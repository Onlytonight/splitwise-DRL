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

        # 初始化堆叠缓冲区 (Deque 会自动挤出旧数据)
        self.state_buffer = deque(
            [np.zeros(self.feature_dim) for _ in range(stack_size)],
            maxlen=stack_size
        )

        # 用于计算速率的累积计数器 (上一时刻的值)
        self.last_stats = {
            'arrival_count': 0,
            'completed_tokens': 0,
            'completed_prompts': 0,
            'kv_transferred_bytes': 0
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
        snapshot.extend([rps])

        # 2. Token Generation Rate (Splitwise 核心负载)
        # 假设 application 对象有 total_tokens_generated 属性
        curr_tokens = self.router.total_complete_token
        delta_tokens = curr_tokens - self.last_stats['completed_tokens']
        token_rate = delta_tokens / interval
        # 3. Prompt Generation Rate (Splitwise 核心负载)
        curr_prompts = self.router.total_complete_prompt
        delta_prompts = curr_prompts - self.last_stats['completed_prompts']
        prompt_rate = delta_prompts / interval

        snapshot.extend([prompt_rate,token_rate])

        # 3 & 4. Avg Input/Output Length (从 router 的最近请求中获取)
        avg_prompt_len,avg_output_len,arrivals = self.router.get_recent_avg_len()
        assert arrivals == delta_arrivals

        snapshot.extend([ avg_prompt_len, avg_output_len])

        # --- B. 队列特征 (Queue) [3 dim] ---
        # 1. Prompt Queue Length
        # 2. Decoding Queue Length (Splitwise 瓶颈所在)
        # 3. Avg Wait Time
        p_queue, d_queue,wait_time,n_p, n_t, util_mem = self.get_instance_feature()
        snapshot.extend([p_queue, d_queue, wait_time])

        # --- C. 资源状态 (Resources) [5 dim] ---
        # 1-2. 实例数量 (Prompt, Token)，已获得
        # 3. Prefill Utilization (计算平均值)
        # 4. Decoding Utilization
        # 5. KV Cache Memory Utilization (防止 OOM)
        util_p,util_d = self.get_avg_utilization(current_time,interval)

        snapshot.extend([n_p, n_t, util_p,util_d, util_mem])

        # --- D. 瓶颈特征 (Bottleneck) [1 dim] ---
        # 1. Interconnect Bandwidth Util (KV 传输压力)
        # curr_bytes = self.cluster.total_kv_transferred_bytes
        # delta_bytes = curr_bytes - self.last_stats['kv_transferred_bytes']
        # # 假设总带宽已知，计算利用率
        # net_util = (delta_bytes / interval) / self.cluster.TOTAL_BANDWIDTH_CAPACITY
        # snapshot.extend([net_util])

        # --- E. 性能反馈 (SLO) [6 dim] ---
        # 计算区间内的性能指标
        ttft, tbt, vio_slo_rate = self.scheduler.get_period_result()
        # ttft 和 tbt 分别包含 [p50, p90, p99]
        TTFT_SLO = [2, 3, 6]
        TBT_SLO = [1.25, 1.5, 5]

        # 归一化的 TTFT 和 TBT 比率（用于状态表示）
        ttft_rate = []
        tbt_rate = []
        for i in range(len(TTFT_SLO)):
            ttft_rate.append(ttft[i] / TTFT_SLO[i])
            tbt_rate.append(tbt[i] / TBT_SLO[i])

        snapshot.extend(ttft_rate + tbt_rate)

        # --- 更新累积状态供下次使用 ---
        self.last_stats.update({
            'arrival_count': curr_arrivals,
            'completed_tokens': curr_tokens,
            'completed_prompts': curr_prompts,
        })

        # 返回完整的 SLO 统计数据给 Reward 计算器
        # slo_stats 格式：[[ttft_p50, ttft_p90, ttft_p99], [tbt_p50, tbt_p90, tbt_p99], [ttft_vio, tbt_vio]]
        # slo_stats = [ttft, tbt, vio_slo_rate]

        return (np.array(snapshot, dtype=np.float32), [n_p, n_t, util_p, util_d],
                [prompt_rate,token_rate,p_queue, d_queue,n_p,n_t], rps)

    def _normalize(self, raw_vector):
        """
        归一化处理：Log-scaling 用于长尾分布 (队列、速率)，Min-Max 用于有界值
        使用append方式动态构建归一化向量，便于增删特征
        """
        norm_vec = []

        idx = 0

        # 1. 负载类 (RPS, TokenRate，promptRate) -> Log1p (平滑大数值)
        # Log(x + 1) 可以把 0~10000 压缩到 0~9
        norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)  # RPS
        idx += 1
        norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)  # TokenRate
        idx += 1
        norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)  # TokenRate
        idx += 1

        # Prompt/Output Length -> 除以一个最大常数
        norm_vec.append(np.clip(raw_vector[idx] / 4096.0, 0, 1))  # PromptLen
        idx += 1
        norm_vec.append(np.clip(raw_vector[idx] / 2048.0, 0, 1))  # OutputLen
        idx += 1

        # 2. 队列类 -> Log1p 并缩放
        norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)  # P Queue
        idx += 1
        norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)  # D Queue
        idx += 1
        norm_vec.append(np.tanh(raw_vector[idx] / 10.0))   # Wait time (假设平均10s)
        idx += 1

        # 3. 资源类 (Instance Counts) -> 除以最大集群规模
        MAX_INSTANCES = 100
        norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # n_p
        idx += 1
        norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # n_t
        idx += 1

        # 4. 利用率 (0-1之间，直接使用)
        norm_vec.append(np.clip(raw_vector[idx], 0, 1))  # util_p
        idx += 1
        norm_vec.append(np.clip(raw_vector[idx], 0, 1))  # util_d
        idx += 1
        norm_vec.append(np.clip(raw_vector[idx], 0, 1))  # util_mem
        idx += 1

        # 5. 网络利用率 (0-1之间)
        # norm_vec.append(np.clip(raw_vector[idx], 0, 1))  # net_util
        # idx += 1

        # 6. Performance指标 (TTFT和TBT的归一化比率)
        # 这些是 actual/SLO 的比率，需要特殊处理
        # 比率 < 1 表示满足SLO（好），> 1 表示违反SLO（坏）
        # 使用 tanh 压缩，使得 0-2 的范围映射到 0-1 之间
        for i in range(6):  # 3个TTFT + 3个TBT
            norm_vec.append(np.tanh(raw_vector[idx]))
            idx += 1

        return np.array(norm_vec, dtype=np.float32)

        # rl_utils.py

    def get_state_and_stats(self, current_time, interval):
        # 1. 收集原始快照
        raw_snapshot, instance_num, reward_stats, rps = self._collect_snapshot(current_time, interval)

        # slo_stats 格式：[[ttft_p50, ttft_p90, ttft_p99], [tbt_p50, tbt_p90, tbt_p99], [ttft_vio, tbt_vio]]
        # 这是给 Reward 函数用的完整统计数据

        # 2. 归一化 & 堆叠
        normalized = self._normalize(raw_snapshot)
        self.state_buffer.append(normalized)
        stacked_state = np.concatenate(self.state_buffer)

        return stacked_state, reward_stats, instance_num, rps

    def get_instance_feature(self):
        # 获取第一个应用的调度器
        scheduler = self.scheduler

        # 初始化计数器
        total_memory = 0

        # 获取活跃实例（排除扩缩容中的实例）
        if hasattr(self.applications[0], 'scaling_manager') and \
           self.applications[0].scaling_manager is not None:
            active_instances = self.applications[0].scaling_manager.get_active_instances(scheduler.instances)
        else:
            active_instances = scheduler.instances

        instance_len = len(active_instances) if active_instances else 1

        # 从sch获取
        total_pending_prompt_queue_length,total_pending_tokens,avg_time = scheduler.get_queue_stats()
        for instance in active_instances:
            total_memory += instance.memory

        # 获取各类型实例数量（包括活跃的）
        if hasattr(self.applications[0], 'scaling_manager') and \
           self.applications[0].scaling_manager is not None:
            scaling_manager = self.applications[0].scaling_manager
            n_p = len(scaling_manager.get_active_instances(scheduler.prompt_instances))
            n_t = len(scaling_manager.get_active_instances(scheduler.token_instances))
        else:
            n_p = len(scheduler.prompt_instances)
            n_t = len(scheduler.token_instances)
        # print(f"n_p: {n_p}, n_t: {n_t}")

        return total_pending_prompt_queue_length, total_pending_tokens, avg_time, \
               n_p, n_t, total_memory/instance_len  # Removed n_m, always return 0

    def compute_util(self, instances, current_time, interval):
        """
        计算实例的平均利用率
        通过获取每个实例的累计忙碌时间并除以时间间隔
        调用后会重置每个实例的累计时间计数器
        """
        # 获取活跃实例
        if hasattr(self.applications[0], 'scaling_manager') and \
           self.applications[0].scaling_manager is not None:
            active_instances = self.applications[0].scaling_manager.get_active_instances(instances)
        else:
            active_instances = instances

        if not active_instances:
            return 0.0

        total_util = 0.0
        for instance in active_instances:
            # 获取并重置累计忙碌时间
            busy_time = instance.get_and_reset_busy_time()
            # 计算利用率 = 忙碌时间 / 时间间隔
            utilization = min(busy_time / interval, 1.0) if interval > 0 else 0.0
            total_util += utilization

        # 返回平均利用率
        return total_util / len(active_instances)

    def get_avg_utilization(self,current_time,interval):
        # 获取第一个应用的调度器
        return self.compute_util(self.scheduler.prompt_instances,current_time,interval),\
            self.compute_util(self.scheduler.token_instances,current_time,interval)
            # self.compute_util(self.scheduler.mixed_instances,current_time,interval)

    @property
    def scheduler(self):
        return self.applications[0].scheduler

    @property
    def feature_dim(self):
        # Workload(5) + Queue(3) + Resource(5) + Bottleneck(0) + Performance(6) = 19
        return 19