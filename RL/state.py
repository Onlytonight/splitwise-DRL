import numpy as np
from collections import deque
import logging


class RLStateCollector:
    """
    支持三种模式：
    - mode="joint": 原来的单一 RL agent，观察整个系统（保持兼容）
    - mode="prompt": 只构造与 prompt 池 + TTFT 相关的状态
    - mode="token": 只构造与 token 池 + TBT 相关的状态
    """
    # 类级别的共享状态，用于确保在一次决策周期内 _collect_snapshot 只被调用一次
    _shared_last_stats = {
        'arrival_count': 0,
        'completed_tokens': 0,
        'completed_prompts': 0,
        'kv_transferred_bytes': 0
    }
    _snapshot_cache = None
    _snapshot_cache_time = None
    _snapshot_cache_interval = None

    def __init__(self, cluster, router, applications, stack_size=4, mode: str = "joint"):
        """
        :param cluster: 仿真器的 cluster 对象 (获取机器资源)
        :param router: 仿真器的 router 对象 (获取队列信息)
        :param applications: 应用列表 (获取 SLO 和请求统计)
        :param stack_size: 时间窗堆叠的大小 (默认 4)
        :param mode: "joint" / "prompt" / "token"
        """
        assert mode in ("joint", "prompt", "token")
        self.cluster = cluster
        self.router = router
        self.applications = applications
        self.stack_size = stack_size
        self.mode = mode

        # 初始化堆叠缓冲区 (Deque 会自动挤出旧数据)
        self.state_buffer = deque(
            [np.zeros(self.feature_dim) for _ in range(stack_size)],
            maxlen=stack_size
        )

        # 使用共享的 last_stats，确保 prompt 和 token collector 使用相同的基础数据
        self.last_stats = RLStateCollector._shared_last_stats

    def _collect_snapshot(self, current_time, interval):
        """
        收集原始状态数据（完整 19 维），不同 mode 在后续再做裁剪。
        使用类级别的缓存机制，确保在一次决策周期内只被调用一次。
        """
        # 检查缓存：如果时间戳和间隔相同，说明是同一个决策周期内的调用，直接返回缓存
        if (RLStateCollector._snapshot_cache is not None and 
            RLStateCollector._snapshot_cache_time == current_time and
            RLStateCollector._snapshot_cache_interval == interval):
            return RLStateCollector._snapshot_cache
        
        snapshot = []
        stats = self.last_stats

        # --- A. 负载特征 (Workload) [3 dim] ---
        # 1. RPS (Requests Per Second)
        curr_arrivals = self.router.total_arrivals
        delta_arrivals = curr_arrivals - stats['arrival_count']
        rps = delta_arrivals / interval
        snapshot.extend([rps])


        # 2. Prompt/Token Generation Rate (Splitwise 核心负载)
        curr_tokens = self.router.total_complete_token
        delta_tokens = curr_tokens - stats['completed_tokens']
        token_rate = delta_tokens / interval

        curr_prompts = self.router.total_complete_prompt
        delta_prompts = curr_prompts - stats['completed_prompts']
        prompt_rate = delta_prompts / interval
        snapshot.extend([prompt_rate,token_rate])

        # 3 & 4. Avg Input/Output Length (从 router 的最近请求中获取)
        avg_prompt_len, avg_output_len, arrivals = self.router.get_recent_avg_len()
        snapshot.extend([avg_prompt_len, avg_output_len])

        # --- B. 队列特征 (Queue) [3 dim] ---
        p_queue, d_queue, wait_time, n_p, n_t, util_mem,avg_prompt_size = self.get_instance_feature()
        snapshot.extend([p_queue, d_queue, wait_time])

        # --- C. 资源状态 (Resources) [5 dim] ---
        util_p, util_d = self.get_avg_utilization(current_time, interval)
        snapshot.extend([n_p, n_t, util_p, util_d, util_mem])

        # --- D. 性能反馈 (SLO) [6 dim] ---
        ttft, tbt, vio_slo_rate = self.scheduler.get_period_result()
        TTFT_SLO = [2, 3, 6]
        TBT_SLO = [1.25, 1.5, 5]

        ttft_rate = [ttft[i] / TTFT_SLO[i] for i in range(len(TTFT_SLO))]
        tbt_rate = [tbt[i] / TBT_SLO[i] for i in range(len(TBT_SLO))]

        snapshot.extend(ttft_rate + tbt_rate)

        # --- 更新累积状态供下次使用 ---
        self.last_stats.update({
            'arrival_count':curr_arrivals,
            'completed_tokens': curr_tokens,
            'completed_prompts': curr_prompts,
        })



        # reward_stats: 保持原有语义，给奖励函数使用的"快速指标"

        reward_stats = [prompt_rate, token_rate, p_queue, d_queue, n_p, n_t,avg_prompt_size]
        instance_num = [n_p, n_t, util_p, util_d]

        result = (np.array(snapshot, dtype=np.float32), instance_num, reward_stats, rps)
        
        # 缓存结果，供同一决策周期内的其他 collector 使用
        RLStateCollector._snapshot_cache = result
        RLStateCollector._snapshot_cache_time = current_time
        RLStateCollector._snapshot_cache_interval = interval

        return result

    # ------------------------- 归一化与特征裁剪 ------------------------- #
    def _normalize_joint(self, raw_vector):
        """
        原有 19 维 joint 状态的归一化，保持兼容。
        """
        norm_vec = []
        idx = 0

        # Workload: RPS, prompt_rate, token_rate
        norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)  # RPS
        idx += 1
        norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)  # prompt_rate
        idx += 1
        norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)  # token_rate
        idx += 1

        # Prompt/Output Length
        norm_vec.append(np.clip(raw_vector[idx] / 4096.0, 0, 1))  # PromptLen
        idx += 1
        norm_vec.append(np.clip(raw_vector[idx] / 2048.0, 0, 1))  # OutputLen
        idx += 1

        # Queue
        norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)  # P Queue
        idx += 1
        norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)  # D Queue
        idx += 1
        norm_vec.append(np.tanh(raw_vector[idx] / 10.0))   # Wait time
        idx += 1

        # Resource
        MAX_INSTANCES = 100
        norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # n_p
        idx += 1
        norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # n_t
        idx += 1

        norm_vec.append(np.clip(raw_vector[idx], 0, 1))  # util_p
        idx += 1
        norm_vec.append(np.clip(raw_vector[idx], 0, 1))  # util_d
        idx += 1
        norm_vec.append(np.clip(raw_vector[idx], 0, 1))  # util_mem
        idx += 1

        # TTFT/TBT rates
        for _ in range(6):
            norm_vec.append(np.tanh(raw_vector[idx]))
            idx += 1

        return np.array(norm_vec, dtype=np.float32)

    def _build_prompt_raw(self, full_vector):
        """
        从 19 维 full_vector 中挑选跟 prompt 池 + TTFT 强相关的特征。
        索引含义参考 _collect_snapshot。
        """
        # 下标对照：
        # 0:rps, 1:prompt_rate, 2:token_rate,
        # 3:prompt_len, 4:output_len,
        # 5:p_queue, 6:d_queue, 7:wait,
        # 8:n_p, 9:n_t, 10:util_p,11:util_d,12:util_mem,
        # 13-15:ttft_rate, 16-18:tbt_rate
        idxs = [
            0,   # rps
            1,   # prompt_rate
            3,   # prompt_len
            5,   # prompt queue
            7,   # wait time
            8,   # n_p
            10,  # util_p
            12,  # util_mem
            13, 14, 15  # TTFT ratios
        ]
        return full_vector[idxs]

    def _build_token_raw(self, full_vector):
        """
        从 19 维 full_vector 中挑选跟 token 池 + TBT 强相关的特征。
        """
        idxs = [
            0,   # rps
            2,   # token_rate
            4,   # output_len
            6,   # decoding queue
            7,   # wait time
            9,   # n_t
            11,  # util_d
            12,  # util_mem
            16, 17, 18  # TBT ratios
        ]
        return full_vector[idxs]

    def _normalize_prompt(self, raw_vec):
        norm = []
        idx = 0

        # rps, prompt_rate
        norm.append(np.log1p(raw_vec[idx]) / 10.0)
        idx += 1
        norm.append(np.log1p(raw_vec[idx]) / 10.0)
        idx += 1

        # prompt_len
        norm.append(np.clip(raw_vec[idx] / 4096.0, 0, 1))
        idx += 1

        # p_queue
        norm.append(np.log1p(raw_vec[idx]) / 10.0)
        idx += 1

        # wait_time
        norm.append(np.tanh(raw_vec[idx] / 10.0))
        idx += 1

        # n_p
        MAX_INSTANCES = 100
        norm.append(np.clip(raw_vec[idx] / MAX_INSTANCES, 0, 1))
        idx += 1

        # util_p
        norm.append(np.clip(raw_vec[idx], 0, 1))
        idx += 1

        # mem util
        norm.append(np.clip(raw_vec[idx], 0, 1))
        idx += 1

        # TTFT ratios
        for _ in range(3):
            norm.append(np.tanh(raw_vec[idx]))
            idx += 1

        return np.array(norm, dtype=np.float32)

    def _normalize_token(self, raw_vec):
        norm = []
        idx = 0

        # rps, token_rate
        norm.append(np.log1p(raw_vec[idx]) / 10.0)
        idx += 1
        norm.append(np.log1p(raw_vec[idx]) / 10.0)
        idx += 1

        # output_len
        norm.append(np.clip(raw_vec[idx] / 2048.0, 0, 1))
        idx += 1

        # d_queue
        norm.append(np.log1p(raw_vec[idx]) / 10.0)
        idx += 1

        # wait_time
        norm.append(np.tanh(raw_vec[idx] / 10.0))
        idx += 1

        # n_t
        MAX_INSTANCES = 100
        norm.append(np.clip(raw_vec[idx] / MAX_INSTANCES, 0, 1))
        idx += 1

        # util_d
        norm.append(np.clip(raw_vec[idx], 0, 1))
        idx += 1

        # mem util
        norm.append(np.clip(raw_vec[idx], 0, 1))
        idx += 1

        # TBT ratios
        for _ in range(3):
            norm.append(np.tanh(raw_vec[idx]))
            idx += 1

        return np.array(norm, dtype=np.float32)

    def get_state_and_stats(self, current_time, interval):
        """
        统一接口，根据 mode 返回对应的堆叠状态：
        - joint: 19 维 * stack_size
        - prompt/token: 11 维 * stack_size
        """
        full_snapshot, instance_num, reward_stats, rps = self._collect_snapshot(current_time, interval)

        if self.mode == "joint":
            normalized = self._normalize_joint(full_snapshot)
        elif self.mode == "prompt":
            prompt_raw = self._build_prompt_raw(full_snapshot)
            normalized = self._normalize_prompt(prompt_raw)
        else:  # "token"
            token_raw = self._build_token_raw(full_snapshot)
            normalized = self._normalize_token(token_raw)

        self.state_buffer.append(normalized)
        stacked_state = np.concatenate(self.state_buffer)

        return stacked_state, reward_stats, instance_num, rps

    @classmethod
    def clear_snapshot_cache(cls):
        """
        清除快照缓存，应该在每次决策周期结束时调用。
        确保下一个决策周期会重新收集快照。
        """
        cls._snapshot_cache = None
        cls._snapshot_cache_time = None
        cls._snapshot_cache_interval = None

    def get_instance_feature(self):
        # 获取第一个应用的调度器
        scheduler = self.scheduler

        # 从 scheduler 获取队列统计
        total_pending_prompt_queue_length, total_pending_tokens, avg_time,avg_prompt_size = scheduler.get_queue_stats()

        # 获取各类型实例数量（包括活跃的）
        if hasattr(self.applications[0], 'scaling_manager') and \
           self.applications[0].scaling_manager is not None:
            scaling_manager = self.applications[0].scaling_manager
            active_prompts = scaling_manager.get_active_instances(scheduler.prompt_instances)
            active_tokens = scaling_manager.get_active_instances(scheduler.token_instances)
        else:
            scaling_manager = None
            active_prompts = scheduler.prompt_instances
            active_tokens = scheduler.token_instances

        n_p = len(active_prompts)
        n_t = len(active_tokens)
        # prompt_instance_queue_len = sum(len(i.pending_queue) for i in active_prompts)
        # tokens_instance_queue_len = sum(len(i.pending_queue) for i in active_tokens)
        # print("实例堆积状态:",prompt_instance_queue_len,tokens_instance_queue_len)
        # print("sch堆积状态:",total_pending_prompt_queue_length,total_pending_tokens)

        # -------- util_mem：根据当前 collector 的 mode，分别按 prompt/token 计算 --------
        if self.mode == "prompt":
            target_instances = active_prompts
        elif self.mode == "token":
            target_instances = active_tokens
        else:  # joint，保持原来对所有实例求平均的语义
            if scaling_manager is not None:
                target_instances = scaling_manager.get_active_instances(scheduler.instances)
            else:
                target_instances = scheduler.instances

        if target_instances:
            total_memory = sum(inst.memory for inst in target_instances)
            util_mem = total_memory / len(target_instances)
        else:
            util_mem = 0.0

        return (
            total_pending_prompt_queue_length,
            total_pending_tokens,
            avg_time,
            n_p,
            n_t,
            util_mem,
            avg_prompt_size
        )

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
        """
        根据模式返回不同的单步特征维度：
        - joint: 19
        - prompt: 11
        - token: 11
        """
        if self.mode == "joint":
            return 19
        else:
            return 11