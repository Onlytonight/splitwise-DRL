import numpy as np
from collections import deque
import logging

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
        # Workload(4) + Queue(3) + Resource(6) + Bottleneck(1) + Performance(6) = 20
        # Performance包含3个TTFT指标和3个TBT指标
        self.feature_dim = 20
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

        return np.array(snapshot, dtype=np.float32),[n_p, n_t, n_m,util_p,util_d,net_util],vio_slo_rate,rps

    def _normalize(self, raw_vector):
        """
        归一化处理：Log-scaling 用于长尾分布 (队列、速率)，Min-Max 用于有界值
        特征顺序：
        [0-3]: Workload (RPS, TokenRate, PromptLen, OutputLen)
        [4-6]: Queue (PQueue, DQueue, WaitTime)
        [7-9]: Instance Counts (n_p, n_t, n_m)
        [10-12]: Utilization (util_p, util_d, util_mem)
        [13]: Network Util
        [14-19]: Performance (3个TTFT + 3个TBT)
        """
        norm_vec = np.zeros_like(raw_vector)

        # 1. 负载类 (RPS, TokenRate) -> Log1p (平滑大数值)
        # Log(x + 1) 可以把 0~10000 压缩到 0~9
        norm_vec[0] = np.log1p(raw_vector[0]) / 10.0  # 进一步缩放到合理范围
        norm_vec[1] = np.log1p(raw_vector[1]) / 10.0
        # Prompt/Output Length -> 除以一个最大常数
        norm_vec[2] = np.clip(raw_vector[2] / 4096.0, 0, 1)
        norm_vec[3] = np.clip(raw_vector[3] / 2048.0, 0, 1)

        # 2. 队列类 -> Log1p 并缩放
        norm_vec[4] = np.log1p(raw_vector[4]) / 10.0  # P Queue
        norm_vec[5] = np.log1p(raw_vector[5]) / 10.0  # D Queue
        norm_vec[6] = np.tanh(raw_vector[6] / 10.0)   # Wait time (假设平均10s)

        # 3. 资源类 (Instance Counts) -> 除以最大集群规模
        MAX_INSTANCES = 100
        norm_vec[7] = np.clip(raw_vector[7] / MAX_INSTANCES, 0, 1)
        norm_vec[8] = np.clip(raw_vector[8] / MAX_INSTANCES, 0, 1)
        norm_vec[9] = np.clip(raw_vector[9] / MAX_INSTANCES, 0, 1)

        # 4. 利用率 (0-1之间，直接使用)
        norm_vec[10] = np.clip(raw_vector[10], 0, 1)  # util_p
        norm_vec[11] = np.clip(raw_vector[11], 0, 1)  # util_d
        norm_vec[12] = np.clip(raw_vector[12], 0, 1)  # util_mem

        # 5. 网络利用率 (0-1之间)
        norm_vec[13] = np.clip(raw_vector[13], 0, 1)  # net_util

        # 6. Performance指标 (TTFT和TBT的归一化比率)
        # 这些是 actual/SLO 的比率，需要特殊处理
        # 比率 < 1 表示满足SLO（好），> 1 表示违反SLO（坏）
        # 使用 tanh 压缩，使得 0-2 的范围映射到 0-1 之间
        for i in range(14, 20):
            norm_vec[i] = np.tanh(raw_vector[i])

        return norm_vec

        # rl_utils.py

    def get_state_and_stats(self, current_time, interval):
        # 1. 收集原始快照
        raw_snapshot,instance_num,vio_SLO,rps = self._collect_snapshot(current_time, interval)

        # 从 raw_snapshot 中提取需要的统计量传给 Reward 函数
        # stats = raw_snapshot[14:]

        # 2. 归一化 & 堆叠
        normalized = self._normalize(raw_snapshot)
        self.state_buffer.append(normalized)
        stacked_state = np.concatenate(self.state_buffer)

        return stacked_state, vio_SLO,instance_num,rps

    def get_instance_feature(self):
        # 获取第一个应用的调度器
        scheduler = self.scheduler
        
        # 初始化计数器
        total_pending_prompt_queue_length = 0
        total_pending_tokens = 0
        total_time = 0
        total_memory = 0
        
        # 获取活跃实例（排除扩缩容中的实例）
        if hasattr(self.applications[0], 'scaling_manager') and \
           self.applications[0].scaling_manager is not None:
            active_instances = self.applications[0].scaling_manager.get_active_instances(scheduler.instances)
        else:
            active_instances = scheduler.instances
        
        instance_len = len(active_instances) if active_instances else 1
        
        # 遍历所有活跃实例
        for instance in active_instances:
            # 累加pending_prompt_queue的长度
            if hasattr(instance, 'pending_prompt_queue'):
                total_pending_prompt_queue_length += len(instance.pending_prompt_queue)
            # 累加pending_tokens
            if hasattr(instance, 'token_queue_size'):
                total_pending_tokens += instance.token_queue_size
            if hasattr(instance, 'get_waiting_tasks_info'):
                total_time += instance.get_waiting_tasks_info()
            total_memory += instance.memory
        
        # 获取各类型实例数量（包括活跃的）
        if hasattr(self.applications[0], 'scaling_manager') and \
           self.applications[0].scaling_manager is not None:
            scaling_manager = self.applications[0].scaling_manager
            n_p = len(scaling_manager.get_active_instances(scheduler.prompt_instances))
            n_t = len(scaling_manager.get_active_instances(scheduler.token_instances))
            n_m = len(scaling_manager.get_active_instances(
                getattr(scheduler, 'mixed_instances', [])))
        else:
            n_p = len(scheduler.prompt_instances)
            n_t = len(scheduler.token_instances)
            n_m = len(getattr(scheduler, 'mixed_instances', []))
            
        return total_pending_prompt_queue_length, total_pending_tokens, total_time/instance_len, \
               n_p, n_t, n_m, total_memory/instance_len

    def compute_util(self, instances, current_time, interval):
        # 不确定逻辑对不对
        # 如果上次有任务完成距离现在>10,则该interval一直空闲；如果上次有任务完成超过现在，则一直繁忙；
        
        # 获取活跃实例
        if hasattr(self.applications[0], 'scaling_manager') and \
           self.applications[0].scaling_manager is not None:
            active_instances = self.applications[0].scaling_manager.get_active_instances(instances)
        else:
            active_instances = instances
        
        if not active_instances:
            return 0.0
        
        util = 0
        for instance in active_instances:
            if not hasattr(instance, 'last_complete_time'):
                continue
            
            if current_time - instance.last_complete_time > interval:
                busy_time = 0
            elif current_time - instance.last_complete_time < 0:
                busy_time = 1
            else:
                busy_time = (current_time - instance.last_complete_time) / interval
            util += busy_time
        
        return util / len(active_instances)

    def get_avg_utilization(self,current_time,interval):
        # 获取第一个应用的调度器
        return self.compute_util(self.scheduler.prompt_instances,current_time,interval),\
            self.compute_util(self.scheduler.token_instances,current_time,interval),\
            self.compute_util(self.scheduler.mixed_instances,current_time,interval)



