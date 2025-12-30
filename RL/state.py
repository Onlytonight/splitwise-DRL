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

    # 共享特征：prompt 和 token 模式都需要的特征
    # 修改这里会影响 prompt 和 token 两个模式
    _SHARED_FEATURES = {
        "needs_rps": True,
        "needs_wait_time": True,
        "needs_util_mem": True,
        "needs_last_action": True,  # 上一次动作特征
    }
    
    # 共享特征的名称映射，用于通过 enabled_features 控制
    _SHARED_FEATURE_NAMES = {
        "rps": "needs_rps",
        "wait_time": "needs_wait_time",
        "util_mem": "needs_util_mem",
        "action": "needs_last_action",  # 或 "last_action"
        "last_action": "needs_last_action",  # 支持两种名称
    }

    _FEATURE_PAIRS = {
        "rate": ("needs_prompt_rate", "needs_token_rate", True),
        "length": ("needs_prompt_len", "needs_output_len", True),
        "queue": ("needs_p_queue", "needs_d_queue", True),
        "instance_count": ("needs_n_p", "needs_n_t", True),
        "utilization": ("needs_util_p", "needs_util_d", True),
        "slo": ("needs_ttft", "needs_tbt", True),
        "scaling": ("needs_scaling_prompt", "needs_scaling_token", True),
        "none_count": ("needs_prompt_none_count", "needs_token_none_count", True),
    }
    
    # Joint 模式的完整配置（所有特征都启用）
    _JOINT_CONFIG = {
        "needs_rps": True,
        "needs_prompt_rate": True,
        "needs_token_rate": True,
        "needs_prompt_len": True,
        "needs_output_len": True,
        "needs_p_queue": True,
        "needs_d_queue": True,
        "needs_wait_time": True,
        "needs_n_p": True,
        "needs_n_t": True,
        "needs_util_p": True,
        "needs_util_d": True,
        "needs_util_mem": True,
        "needs_ttft": True,
        "needs_tbt": True,
        "needs_scaling_prompt": True,
        "needs_scaling_token": True,
        "needs_last_action": True,
        "needs_prompt_none_count": True,
        "needs_token_none_count": True,
    }
    
    @classmethod
    def _build_feature_config(cls, mode, enabled_features=None):
        """
        根据模式构建特征配置。
        对于 prompt/token 模式，使用特征对机制，确保对应特征同步。
        
        :param mode: "joint" / "prompt" / "token"
        :param enabled_features: 可选，要启用的特征名称列表
                                 可以包含特征对名称（如 "instance_count", "scaling"）
                                 也可以包含共享特征名称（如 "rps", "wait_time", "util_mem"）
                                 如果为 None，则使用默认配置（所有特征都启用）
        :return: 特征配置字典
        """
        if mode == "joint":
            config = cls._JOINT_CONFIG.copy()
            # 如果指定了 enabled_features，则只启用指定的特征
            if enabled_features is not None:
                enabled_set = set(enabled_features)
                # 禁用所有共享特征
                for key in cls._SHARED_FEATURES:
                    config[key] = False
                # 禁用所有特征对
                for pair_name in cls._FEATURE_PAIRS:
                    prompt_feat, token_feat, _ = cls._FEATURE_PAIRS[pair_name]
                    config[prompt_feat] = False
                    config[token_feat] = False
                
                # 启用指定的共享特征
                for shared_name, feature_key in cls._SHARED_FEATURE_NAMES.items():
                    if shared_name in enabled_set:
                        config[feature_key] = True
                
                # 启用指定的特征对
                for pair_name in enabled_set:
                    if pair_name in cls._FEATURE_PAIRS:
                        prompt_feat, token_feat, _ = cls._FEATURE_PAIRS[pair_name]
                        config[prompt_feat] = True
                        config[token_feat] = True
            return config
        
        # 从共享特征开始
        config = {}
        
        # 如果指定了 enabled_features，则根据列表启用共享特征
        if enabled_features is not None:
            enabled_set = set(enabled_features)
            # 只启用 enabled_features 中指定的共享特征
            for shared_name, feature_key in cls._SHARED_FEATURE_NAMES.items():
                config[feature_key] = shared_name in enabled_set
        else:
            # 默认启用所有共享特征
            config = cls._SHARED_FEATURES.copy()
        
        # 添加特征对
        for pair_name, (prompt_feat, token_feat, default_enabled) in cls._FEATURE_PAIRS.items():
            # 如果指定了 enabled_features，则只启用列表中的特征对
            if enabled_features is not None:
                enabled = pair_name in enabled_set
            else:
                enabled = default_enabled
            
            if mode == "prompt":
                config[prompt_feat] = enabled
                config[token_feat] = False
            elif mode == "token":
                config[prompt_feat] = False
                config[token_feat] = enabled
        
        return config
    
    @classmethod
    def _get_feature_config(cls, mode, enabled_features=None):
        """
        获取指定模式的特征配置。
        使用此方法而不是直接访问 _FEATURE_CONFIG，确保特征对同步。
        
        :param mode: "joint" / "prompt" / "token"
        :param enabled_features: 可选，要启用的特征名称列表
                                 可以包含特征对名称（如 "instance_count", "scaling"）
                                 也可以包含共享特征名称（如 "rps", "wait_time", "util_mem"）
        :return: 特征配置字典
        """
        return cls._build_feature_config(mode, enabled_features)

    def __init__(self, cluster, router, applications, stack_size=4, mode: str = "joint", reset_shared_stats=False, enabled_features=None):
        """
        :param cluster: 仿真器的 cluster 对象 (获取机器资源)
        :param router: 仿真器的 router 对象 (获取队列信息)
        :param applications: 应用列表 (获取 SLO 和请求统计)
        :param stack_size: 时间窗堆叠的大小 (默认 4)
        :param mode: "joint" / "prompt" / "token"
        :param reset_shared_stats: 是否重置共享的统计状态（用于新 trace 开始时）
        :param enabled_features: 可选，要启用的特征名称列表
                                 如果为 None，则使用默认配置（所有特征都启用）
                                 可用的特征对名称：rate, length, queue, instance_count, utilization, slo, scaling, none_count
                                 可用的共享特征名称：rps, wait_time, util_mem, action (或 last_action)
                                 注意：none_count 特征对包含 prompt_none_count 和 token_none_count
        """
        assert mode in ("joint", "prompt", "token")
        
        self.cluster = cluster
        self.router = router
        self.applications = applications
        self.stack_size = stack_size
        self.mode = mode
        
        # 根据模式构建特征配置（使用特征对机制，确保 prompt/token 同步）
        self.feature_config = self._get_feature_config(mode, enabled_features)

        # 初始化堆叠缓冲区 (Deque 会自动挤出旧数据)
        self.state_buffer = deque(
            [np.zeros(self.feature_dim) for _ in range(stack_size)],
            maxlen=stack_size
        )

        # 使用共享的 last_stats，确保 prompt 和 token collector 使用相同的基础数据
        # 如果指定重置共享状态，则重置类级别的共享字典
        if reset_shared_stats:
            self.last_stats ={
                'arrival_count': 0,
                'completed_tokens': 0,
                'completed_prompts': 0,
                'kv_transferred_bytes': 0
            }
        else:
            self.last_stats = RLStateCollector._shared_last_stats


    def _collect_snapshot(self, current_time, interval):
        """
        根据初始化时选择的特征配置，只收集需要的原始状态数据。
        使用类级别的缓存机制，确保在一次决策周期内只被调用一次。
        """
        # 检查缓存：如果时间戳和间隔相同，说明是同一个决策周期内的调用，直接返回缓存
        if (RLStateCollector._snapshot_cache is not None and 
            RLStateCollector._snapshot_cache_time == current_time and
            RLStateCollector._snapshot_cache_interval == interval):
            return RLStateCollector._snapshot_cache
        
        snapshot = []
        stats = self.last_stats
        cfg = self.feature_config

        # --- A. 负载特征 (Workload) ---
        # 1. RPS (Requests Per Second)
        curr_arrivals = self.router.total_arrivals
        delta_arrivals = curr_arrivals - stats['arrival_count']
        rps = delta_arrivals / interval if interval > 0 else 0.0
        if cfg["needs_rps"]:
            snapshot.append(rps)

        # 2. Prompt/Token Generation Rate
        curr_tokens = self.router.total_complete_token
        curr_prompts = self.router.total_complete_prompt
        
        # 无论特征配置如何，都需要计算这些速率用于奖励计算
        delta_tokens = curr_tokens - stats['completed_tokens']
        token_rate = delta_tokens / interval if interval > 0 else 0.0

        delta_prompts = curr_prompts - stats['completed_prompts']
        prompt_rate = delta_prompts / interval if interval > 0 else 0.0

        if cfg["needs_prompt_rate"]:
            snapshot.append(prompt_rate)
        if cfg["needs_token_rate"]:
            snapshot.append(token_rate)

        # 3 & 4. Avg Input/Output Length
        avg_prompt_len, avg_output_len, arrivals = self.router.get_recent_avg_len()

        if cfg["needs_prompt_len"]:
            snapshot.append(avg_prompt_len)
        if cfg["needs_output_len"]:
            snapshot.append(avg_output_len)

        # --- B. 队列特征 (Queue) ---
        sch_p_queue_tokens, sch_d_queue_tokens, wait_time, avg_prompt_size = self.get_scheduler_feature()
        n_p, n_t, util_mem, ins_p_queue, ins_d_queue = self.get_instance_feature()
        
        if cfg["needs_p_queue"]:
            snapshot.append(sch_p_queue_tokens)
        if cfg["needs_d_queue"]:
            snapshot.append(sch_d_queue_tokens)
        if cfg["needs_wait_time"]:
            snapshot.append(wait_time)

        # --- C. 资源状态 (Resources) ---
        # 无论特征配置如何，都需要计算利用率用于奖励计算
        util_p, util_d = self.get_avg_utilization(current_time, interval)

        if cfg["needs_n_p"]:
            snapshot.append(n_p)
        if cfg["needs_n_t"]:
            snapshot.append(n_t)
        if cfg["needs_util_p"]:
            snapshot.append(util_p)
        if cfg["needs_util_d"]:
            snapshot.append(util_d)
        if cfg["needs_util_mem"]:
            snapshot.append(util_mem)

        # --- D. 性能反馈 (SLO) ---
        # 无论特征配置如何，都需要获取真实的SLO数据用于奖励计算
        ttft, tbt, vio_slo_rate, avg_queue_time, avg_nth_token_overhead, prompt_none_count, token_none_count = self.scheduler.get_period_result()
        # print(prompt_none_count,token_none_count)
        TTFT_SLO = [2, 3, 6]
        TBT_SLO = [1.25, 1.5, 5]

        # 计算真实的TTFT和TBT比率（用于奖励计算）
        ttft_rate = [ttft[i] / TTFT_SLO[i] for i in range(len(TTFT_SLO))]
        tbt_rate = [tbt[i] / TBT_SLO[i] for i in range(len(TBT_SLO))]

        # 只有在特征配置需要时才添加到snapshot
        if cfg["needs_ttft"]:
            snapshot.extend(ttft_rate)
        if cfg["needs_tbt"]:
            snapshot.extend(tbt_rate)

        # --- E. 扩缩容特征 ---
        scaling_up_prompt, scaling_up_token, draining_prompt, draining_token = self.get_scaling_manager_feature()

        if cfg["needs_scaling_prompt"]:
            snapshot.append(scaling_up_prompt)
            snapshot.append(draining_prompt)
        if cfg["needs_scaling_token"]:
            snapshot.append(scaling_up_token)
            snapshot.append(draining_token)

        # --- F. None计数特征（瓶颈指标）---
        if cfg["needs_prompt_none_count"]:
            snapshot.append(prompt_none_count)
        if cfg["needs_token_none_count"]:
            snapshot.append(token_none_count)

        # --- G. 上一次动作特征 ---
        # 将上一次的动作添加到状态中，帮助 agent 了解自己之前做了什么决策
        # 根据配置决定是否包含
        if cfg.get("needs_last_action", False):
            snapshot.append(self.last_action)

        # --- 更新累积状态供下次使用 ---
        # 无论是否需要这些特征，都需要更新统计值，因为可能被其他 collector 使用
        self.last_stats.update({
            'arrival_count': curr_arrivals,
            'completed_tokens': curr_tokens,
            'completed_prompts': curr_prompts,
        })

        # reward_stats: 保持原有语义，给奖励函数使用的"快速指标"
        reward_stats = [prompt_rate, token_rate, sch_p_queue_tokens, sch_d_queue_tokens, n_p, n_t, avg_prompt_size, ttft_rate, tbt_rate, ins_p_queue, ins_d_queue, avg_queue_time, avg_nth_token_overhead]
        instance_num = [n_p, n_t, util_p, util_d]

        result = (np.array(snapshot, dtype=np.float32), instance_num, reward_stats, rps)
        
        # 缓存结果，供同一决策周期内的其他 collector 使用
        RLStateCollector._snapshot_cache = result
        RLStateCollector._snapshot_cache_time = current_time
        RLStateCollector._snapshot_cache_interval = interval

        return result

    # ------------------------- 归一化 ------------------------- #
    def _normalize(self, raw_vector):
        """
        根据模式对收集的原始特征进行归一化。
        由于 _collect_snapshot 已经按模式只收集需要的特征，这里直接按顺序归一化即可。
        """
        norm_vec = []
        idx = 0
        cfg = self.feature_config
        MAX_INSTANCES = 100

        # Workload features
        if cfg["needs_rps"]:
            norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)
            idx += 1
        if cfg["needs_prompt_rate"]:
            norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)
            idx += 1
        if cfg["needs_token_rate"]:
            norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)
            idx += 1

        # Length features
        if cfg["needs_prompt_len"]:
            norm_vec.append(np.clip(raw_vector[idx] / 4096.0, 0, 1))
            idx += 1
        if cfg["needs_output_len"]:
            norm_vec.append(np.clip(raw_vector[idx] / 2048.0, 0, 1))
            idx += 1

        # Queue features
        if cfg["needs_p_queue"]:
            norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)
            idx += 1
        if cfg["needs_d_queue"]:
            norm_vec.append(np.log1p(raw_vector[idx]) / 10.0)
            idx += 1
        if cfg["needs_wait_time"]:
            norm_vec.append(np.tanh(raw_vector[idx] / 10.0))
            idx += 1

        # Resource features
        if cfg["needs_n_p"]:
            norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))
            idx += 1
        if cfg["needs_n_t"]:
            norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))
            idx += 1
        if cfg["needs_util_p"]:
            norm_vec.append(np.clip(raw_vector[idx], 0, 1))
            idx += 1
        if cfg["needs_util_d"]:
            norm_vec.append(np.clip(raw_vector[idx], 0, 1))
            idx += 1
        if cfg["needs_util_mem"]:
            norm_vec.append(np.clip(raw_vector[idx], 0, 1))
            idx += 1

        # SLO features
        if cfg["needs_ttft"]:
            for _ in range(3):
                norm_vec.append(np.tanh(raw_vector[idx]))
                idx += 1
        if cfg["needs_tbt"]:
            for _ in range(3):
                norm_vec.append(np.tanh(raw_vector[idx]))
                idx += 1

        # Scaling features
        if cfg["needs_scaling_prompt"]:
            norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # scaling_up_prompt
            idx += 1
            norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # draining_prompt
            idx += 1
        if cfg["needs_scaling_token"]:
            norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # scaling_up_token
            idx += 1
            norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # draining_token
            idx += 1

        # None count features (瓶颈指标，使用log1p归一化)
        if cfg["needs_prompt_none_count"]:
            norm_vec.append(np.log1p(raw_vector[idx]) / 500.0)  # prompt_none_count
            idx += 1
        if cfg["needs_token_none_count"]:
            norm_vec.append(np.log1p(raw_vector[idx]) / 500.0)  # token_none_count
            idx += 1

        # Last action feature (动作通常在 [-1, 1] 范围内，使用 tanh 归一化到 [-1, 1])
        # 根据配置决定是否归一化
        if cfg.get("needs_last_action", False):
            norm_vec.append(np.tanh(raw_vector[idx]))  # last_action
            idx += 1

        return np.array(norm_vec, dtype=np.float32)

    def get_state_and_stats(self, current_time, interval, last_action=None):
        """
        统一接口，根据 mode 返回对应的堆叠状态（包含上一次动作）：
        - joint: (原维度 + 1) * stack_size（+1 是上一次动作）
        - prompt: (原维度 + 1) * stack_size（+1 是上一次动作）
        - token: (原维度 + 1) * stack_size（+1 是上一次动作）
        
        :param last_action: 上一次执行的动作值（可选），如果提供则更新 last_action
        """
        # 如果提供了上一次动作，更新它
        if last_action is not None:
            self.last_action = float(last_action)
        
        raw_snapshot, instance_num, reward_stats, rps = self._collect_snapshot(current_time, interval)
        normalized = self._normalize(raw_snapshot)

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

    def get_scheduler_feature(self):
        """
        获取调度器（scheduler）相关的特征
        
        Returns:
            tuple: (sch_p_queue_tokens, sch_d_queue_tokens, wait_time, avg_prompt_size)
                - sch_p_queue_tokens: 调度器中待处理的 prompt 队列长度（token 数）
                - sch_d_queue_tokens: 调度器中待处理的 token 队列长度（token 数）
                - wait_time: 平均等待时间
                - avg_prompt_size: 平均 prompt 大小
        """
        scheduler = self.scheduler
        total_pending_prompt_queue_length, total_pending_tokens, avg_time, avg_prompt_size = scheduler.get_queue_stats()
        # print("sch堆积状态:",total_pending_prompt_queue_length,total_pending_tokens)

        return (
            total_pending_prompt_queue_length,
            total_pending_tokens,
            avg_time,
            avg_prompt_size
        )

    def get_instance_feature(self):
        """
        获取实例（instance）相关的特征
        
        Returns:
            tuple: (n_p, n_t, util_mem, ins_p_queue, ins_d_queue)
                - n_p: 活跃的 prompt 实例数量
                - n_t: 活跃的 token 实例数量
                - util_mem: 平均内存利用率（根据 mode 计算不同的实例集合）
                - ins_p_queue: prompt 实例队列总长度
                - ins_d_queue: token 实例队列总长度
        """
        scheduler = self.scheduler

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
        prompt_instance_queue_len = sum(len(i.pending_queue) for i in active_prompts)
        tokens_instance_queue_len = sum(len(i.pending_queue) for i in active_tokens)
        # print("实例堆积状态:",prompt_instance_queue_len,tokens_instance_queue_len)

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
            n_p,
            n_t,
            util_mem,
            prompt_instance_queue_len,
            tokens_instance_queue_len
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

    def get_scaling_manager_feature(self, mode=None):
        """
        获取正在扩缩容的 prompt 和 token 实例数
        
        :param mode: 可选参数，用于兼容旧代码（实际不使用）
        :return: (scaling_up_prompt, scaling_up_token, draining_prompt, draining_token)
        """
        # --- 获取正在扩缩容的 prompt 和 token 实例数 ---
        scaling_up_prompt = 0
        scaling_up_token = 0
        draining_prompt = 0
        draining_token = 0
        
        if hasattr(self.applications[0], 'scaling_manager') and \
           self.applications[0].scaling_manager is not None:
            scaling_manager = self.applications[0].scaling_manager
            
            # 统计正在扩容的 prompt 和 token 实例数
            for instance in scaling_manager.scaling_up_instances:
                instance_tag = scaling_manager.instance_tag.get(
                    instance.instance_id,
                    getattr(instance, 'tag', None)
                )
                if instance_tag == "prompt":
                    scaling_up_prompt += 1
                elif instance_tag == "token":
                    scaling_up_token += 1
            
            # 统计正在缩容（排空）的 prompt 和 token 实例数
            for instance in scaling_manager.draining_instances:
                instance_tag = scaling_manager.instance_tag.get(
                    instance.instance_id,
                    getattr(instance, 'tag', None)
                )
                if instance_tag == "prompt":
                    draining_prompt += 1
                elif instance_tag == "token":
                    draining_token += 1

        return scaling_up_prompt, scaling_up_token, draining_prompt, draining_token

    @property
    def scheduler(self):
        return self.applications[0].scheduler

    @property
    def feature_dim(self):
        """
        根据模式返回不同的单步特征维度：
        - joint: 25 (rps + prompt_rate + token_rate + prompt_len + output_len + 
                     p_queue + d_queue + wait_time + n_p + n_t + util_p + util_d + 
                     util_mem + 3*ttft_rate + 3*tbt_rate + 4*scaling + 
                     2*none_count + last_action)
        - prompt: 14 (rps + prompt_rate + prompt_len + p_queue + wait_time + 
                      n_p + util_p + util_mem + 3*ttft_rate + 2*scaling_prompt + 
                      prompt_none_count + last_action)
        - token: 14 (rps + token_rate + output_len + d_queue + wait_time + 
                     n_t + util_d + util_mem + 3*tbt_rate + 2*scaling_token + 
                     token_none_count + last_action)
        """
        cfg = self.feature_config
        dim = 0
        
        # Workload
        if cfg["needs_rps"]:
            dim += 1
        if cfg["needs_prompt_rate"]:
            dim += 1
        if cfg["needs_token_rate"]:
            dim += 1
        
        # Length
        if cfg["needs_prompt_len"]:
            dim += 1
        if cfg["needs_output_len"]:
            dim += 1
        
        # Queue
        if cfg["needs_p_queue"]:
            dim += 1
        if cfg["needs_d_queue"]:
            dim += 1
        if cfg["needs_wait_time"]:
            dim += 1
        
        # Resource
        if cfg["needs_n_p"]:
            dim += 1
        if cfg["needs_n_t"]:
            dim += 1
        if cfg["needs_util_p"]:
            dim += 1
        if cfg["needs_util_d"]:
            dim += 1
        if cfg["needs_util_mem"]:
            dim += 1
        
        # SLO
        if cfg["needs_ttft"]:
            dim += 3
        if cfg["needs_tbt"]:
            dim += 3
        
        # Scaling
        if cfg["needs_scaling_prompt"]:
            dim += 2
        if cfg["needs_scaling_token"]:
            dim += 2
        
        # None count (瓶颈指标)
        if cfg["needs_prompt_none_count"]:
            dim += 1
        if cfg["needs_token_none_count"]:
            dim += 1
        
        # Last action (根据配置决定是否包含)
        if cfg.get("needs_last_action", False):
            dim += 1
        
        return dim