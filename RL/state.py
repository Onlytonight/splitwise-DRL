import numpy as np
from collections import deque
import logging
from simulator import clock

class RLStateCollector:
    def __init__(self, cluster, router, applications, stack_size=4):
        """
        :param cluster: 仿真器的 cluster 对象 (获取机器资源)
        :param router: 仿真器的 router 对象 (获取队列信息)
        :param applications: 应用列表 (获取 SLO 和请求统计)
        :param stack_size: 时间窗堆叠的大小 (默认 4)
        """
        self.cluster = cluster
        self.router = router
        self.applications = applications
        self.stack_size = stack_size

        # 定义单个时间步的特征维度 (根据下文的 collect_snapshot 计算)
        # Workload(4) + Queue(3) + Resource(6) + Bottleneck(1) + Performance(2) = 16
        self.feature_dim = 16
        self.scheduler = self.applications[0].scheduler

        # 初始化堆叠缓冲区 (Deque 会自动挤出旧数据)
        self.state_buffer = deque(
            [np.zeros(self.feature_dim) for _ in range(stack_size)],
            maxlen=stack_size
        )

        # 用于计算速率的累积计数器 (上一时刻的值)
        self.last_stats = {
            'arrival_count': 0,
            'completed_tokens': 0,
            'kv_transferred_bytes': 0,
            'slo_violation_count_ttft': 0,
            'slo_violation_count_tbt': 0,
            'total_reqs_checked': 0
        }

    def get_stacked_state(self, current_time, interval):
        """
        主入口：获取处理好、堆叠好的状态向量
        """
        # 1. 收集当前时刻的原始快照
        raw_snapshot = self._collect_snapshot(current_time, interval)

        # 2. 归一化处理 (非常重要，否则 PPO 难以收敛)
        normalized_snapshot = self._normalize(raw_snapshot)

        # 3. 推入堆叠缓冲区
        self.state_buffer.append(normalized_snapshot)

        # 4. 展平并返回 (Dim: stack_size * feature_dim)
        return np.concatenate(self.state_buffer)

    def _collect_snapshot(self, current_time, interval):
        """
        收集原始状态数据
        """
        snapshot = []

        # --- A. 负载特征 (Workload) [4 dim] ---
        # 1. RPS (Requests Per Second)
        curr_arrivals = self.router.total_arrivals
        delta_arrivals = curr_arrivals - self.last_stats['arrival_count']
        rps = delta_arrivals / interval

        # 2. Token Generation Rate (Splitwise 核心负载)
        # 假设 application 对象有 total_tokens_generated 属性
        curr_tokens = self.router.total_complete_token
        delta_tokens = curr_tokens - self.last_stats['completed_tokens']
        token_rate = delta_tokens / interval

        # 3 & 4. Avg Input/Output Length (从 router 的最近请求中获取)
        avg_prompt_len,avg_output_len,arrivals = self.router.get_recent_avg_len()
        assert arrivals == delta_arrivals

        snapshot.extend([rps, token_rate, avg_prompt_len, avg_output_len])

        # --- B. 队列特征 (Queue) [3 dim] ---
        # 1. Prompt Queue Length
        # 2. Decoding Queue Length (Splitwise 瓶颈所在)
        # 3. Avg Wait Time
        p_queue, d_queue,wait_time,n_p, n_t, n_m, util_mem = self.get_instance_feature()
        snapshot.extend([p_queue, d_queue, wait_time])

        # --- C. 资源状态 (Resources) [6 dim] ---
        # 1-3. 实例数量 (Prompt, Token, Mixed)，已获得
        # 4. Prefill Utilization (计算平均值)
        # 5. Decoding Utilization
        # 6. KV Cache Memory Utilization (防止 OOM)
        util_p,util_d,net_util = self.get_avg_utilization(current_time,interval)

        snapshot.extend([n_p, n_t, n_m, util_p, util_d, util_mem])

        # --- D. 瓶颈特征 (Bottleneck) [1 dim] ---
        # 1. Interconnect Bandwidth Util (KV 传输压力)
        # curr_bytes = self.cluster.total_kv_transferred_bytes
        # delta_bytes = curr_bytes - self.last_stats['kv_transferred_bytes']
        # # 假设总带宽已知，计算利用率
        # net_util = (delta_bytes / interval) / self.cluster.TOTAL_BANDWIDTH_CAPACITY
        snapshot.extend([net_util])

        # --- E. 性能反馈 (SLO) [2 dim] ---
        # 计算区间内的违约率
        ttft_rate = []
        tbt_rate = []
        ttft,tbt,vio_slo_rate = self.scheduler.get_period_result()
        TTFT_SLO=[2,3,6]
        TBT_SLO=[1.25,1.5,5]
        for i in range(len(TTFT_SLO)):
            ttft_rate.append(ttft[i]/TTFT_SLO[i])
            tbt_rate.append(tbt[i]/TBT_SLO[i])


        snapshot.extend(ttft_rate+tbt_rate)

        # --- 更新累积状态供下次使用 ---
        self.last_stats.update({
            'arrival_count': curr_arrivals,
            'completed_tokens': curr_tokens,
        })

        return np.array(snapshot, dtype=np.float32),[n_p, n_t, n_m,util_p,util_d,net_util],vio_slo_rate

    def _normalize(self, raw_vector):
        """
        归一化处理：Log-scaling 用于长尾分布 (队列、速率)，Min-Max 用于有界值
        """
        # 假设 vector 顺序对应 _collect_snapshot 的顺序
        norm_vec = np.zeros_like(raw_vector)

        # 1. 负载类 (RPS, TokenRate) -> Log1p (平滑大数值)
        # Log(x + 1) 可以把 0~10000 压缩到 0~9
        norm_vec[0] = np.log1p(raw_vector[0])
        norm_vec[1] = np.log1p(raw_vector[1])
        # Prompt/Output Length -> 除以一个最大常数 (例如 4096)
        norm_vec[2] = raw_vector[2] / 4096.0
        norm_vec[3] = raw_vector[3] / 2048.0

        # 2. 队列类 -> Log1p
        norm_vec[4] = np.log1p(raw_vector[4])  # P Queue
        norm_vec[5] = np.log1p(raw_vector[5])  # D Queue
        norm_vec[6] = np.tanh(raw_vector[6])  # Wait time 用 tanh 压缩到 0-1

        # 3. 资源类 (Instance Counts) -> 除以最大集群规模 (例如 100)
        MAX_INSTANCES = 100
        norm_vec[7] = raw_vector[7] / MAX_INSTANCES
        norm_vec[8] = raw_vector[8] / MAX_INSTANCES
        norm_vec[9] = raw_vector[9] / MAX_INSTANCES

        # 4. 利用率 & SLO -> 本身就是 0-1 之间，直接保留
        # Utils (10, 11, 12), Net Util (13), SLO (14, 15) + 4，共20
        norm_vec[10:] = raw_vector[10:]

        return norm_vec

        # rl_utils.py

    def get_state_and_stats(self, current_time, interval):
        # 1. 收集原始快照
        raw_snapshot,instance_num,vio_SLO = self._collect_snapshot(current_time, interval)

        # 从 raw_snapshot 中提取需要的统计量传给 Reward 函数
        # stats = raw_snapshot[14:]

        # 2. 归一化 & 堆叠
        normalized = self._normalize(raw_snapshot)
        self.state_buffer.append(normalized)
        stacked_state = np.concatenate(self.state_buffer)

        return stacked_state, vio_SLO,instance_num

    def get_instance_feature(self):
        # 获取第一个应用的调度器
        scheduler = self.scheduler
        
        # 初始化计数器
        total_pending_prompt_queue_length = 0
        total_pending_tokens = 0
        total_time = 0
        total_memory = 0
        instance_len = len(scheduler.instances)
        # 遍历所有实例
        for instance in scheduler.instances:
            # 累加pending_prompt_queue的长度
            total_pending_prompt_queue_length += len(instance.pending_prompt_queue)
            # 累加pending_tokens
            total_pending_tokens += instance.token_queue_size
            total_time += instance.get_waiting_tasks_info()
            total_memory+= instance.memory
            
        return total_pending_prompt_queue_length, total_pending_tokens,total_time/instance_len, \
        len(scheduler.prompt_instances), len(scheduler.token_instances),len(scheduler.mixed_instances),total_memory/instance_len

    def compute_util(self,instances,current_time,interval):
        # 不确定逻辑对不对
        # 如果上次有任务完成距离现在>10,则该interval一直空闲；如果上次有任务完成超过现在，则一直繁忙；
        util = 0
        for instance in instances:
            if current_time - instance.last_complete_time > interval:
                busy_time = 0
            elif current_time - instance.last_complete_time < 0:
                busy_time = 1
            else:
                busy_time = (current_time - instance.last_complete_time)/interval
            util += busy_time
        return util/len(instances)

    def get_avg_utilization(self,current_time,interval):
        # 获取第一个应用的调度器
        return self.compute_util(self.scheduler.prompt_instances,current_time,interval),\
            self.compute_util(self.scheduler.token_instances,current_time,interval),\
            self.compute_util(self.scheduler.mixed_instances,current_time,interval)



