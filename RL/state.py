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
        'kv_transferred_bytes': 0,
        'last_rps': 0.0  # 上一次的 RPS 值（用于计算 RPS 差值）
    }
    # 为每个 mode 维护独立的缓存，避免 prompt 和 token collector 共享错误的缓存
    _snapshot_cache = {}  # 改为字典，key 为 mode
    _snapshot_cache_time = {}  # 改为字典，key 为 mode
    _snapshot_cache_interval = {}  # 改为字典，key 为 mode
    _snapshot_cache_prompt_action = {}  # 改为字典，key 为 mode
    _snapshot_cache_token_action = {}  # 改为字典，key 为 mode

    # 共享特征：prompt 和 token 模式都需要的特征
    # 修改这里会影响 prompt 和 token 两个模式
    _SHARED_FEATURES = {
        "needs_rps": True,
        "needs_rps_delta": True,  # RPS 差值特征（当前 RPS - 上一次 RPS）
        "needs_wait_time": True,
        "needs_util_mem": True,
        "needs_last_action": True,  # 上一次动作特征
        "needs_none_count": True,  # None计数特征（包含 prompt_none_count 和 token_none_count）
        "needs_instance_count": True,  # 实例数量特征（包含 n_p 和 n_t）
        "needs_draining": True,  # 缩容特征（包含 draining_prompt 和 draining_token）
        "needs_usetime": True,  # 实例使用时间特征
        "needs_timestamp": True,  # 时间戳特征
        "needs_prompt_instance_pending_token": True,  # prompt 实例待处理的 token 归一化值
    }
    
    # 共享特征的名称映射，用于通过 enabled_features 控制
    _SHARED_FEATURE_NAMES = {
        "rps": "needs_rps",
        "rps_delta": "needs_rps_delta",  # RPS 差值特征
        "wait_time": "needs_wait_time",
        "util_mem": "needs_util_mem",
        "action": "needs_last_action",  # 或 "last_action"
        "none_count": "needs_none_count",  # None计数特征
        "instance_count": "needs_instance_count",  # 实例数量特征
        "draining": "needs_draining",  # 缩容特征
        "use_time": "needs_usetime",
        "timestamp": "needs_timestamp",  # 时间戳特征
        "p_ins_pending_token": "needs_prompt_instance_pending_token",  # prompt 实例待处理的 token
    }

    _FEATURE_PAIRS = {
        "rate": ("needs_prompt_rate", "needs_token_rate", True),
        "length": ("needs_prompt_len", "needs_output_len", True),
        "queue": ("needs_p_queue", "needs_d_queue", True),
        "utilization": ("needs_util_p", "needs_util_d", True),
        "slo": ("needs_ttft", "needs_tbt", True),
        "scaling": ("needs_scaling_prompt", "needs_scaling_token", True),
    }
    
    # Joint 模式的完整配置（所有特征都启用）
    _JOINT_CONFIG = {
        "needs_rps": True,
        "needs_rps_delta": True,
        "needs_prompt_rate": True,
        "needs_token_rate": True,
        "needs_prompt_len": True,
        "needs_output_len": True,
        "needs_p_queue": True,
        "needs_d_queue": True,
        "needs_wait_time": True,
        "needs_instance_count": True,
        "needs_util_p": True,
        "needs_util_d": True,
        "needs_util_mem": True,
        "needs_ttft": True,
        "needs_tbt": True,
        "needs_scaling_prompt": True,
        "needs_scaling_token": True,
        "needs_last_action": True,
        "needs_none_count": True,
        "needs_draining": True,
        "needs_usetime": True,
        "needs_timestamp": True,
        "needs_prompt_instance_pending_token": True,
    }
    
    @classmethod
    def _build_feature_config(cls, mode, enabled_features=None):
        """
        根据模式构建特征配置。
        对于 prompt/token 模式，使用特征对机制，确保对应特征同步。
        
        :param mode: "joint" / "prompt" / "token"
        :param enabled_features: 可选，要启用的特征名称列表
                                 可以包含特征对名称（如 "scaling"）
                                 也可以包含共享特征名称（如 "rps", "wait_time", "util_mem", "instance_count"）
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
                                 可以包含特征对名称（如 "scaling"）
                                 也可以包含共享特征名称（如 "rps", "wait_time", "util_mem", "instance_count"）
        :return: 特征配置字典
        """
        return cls._build_feature_config(mode, enabled_features)

    def __init__(self, cluster, router, applications, stack_size=4, mode: str = "joint", reset_shared_stats=False, enabled_features=None, debug_features=False):
        """
        :param cluster: 仿真器的 cluster 对象 (获取机器资源)
        :param router: 仿真器的 router 对象 (获取队列信息)
        :param applications: 应用列表 (获取 SLO 和请求统计)
        :param stack_size: 时间窗堆叠的大小 (默认 4)
        :param mode: "joint" / "prompt" / "token"
        :param reset_shared_stats: 是否重置共享的统计状态（用于新 trace 开始时）
        :param enabled_features: 可选，要启用的特征名称列表
                                 如果为 None，则使用默认配置（所有特征都启用）
                                 可用的特征对名称：rate, length, queue, utilization, slo, scaling
                                 可用的共享特征名称：rps, wait_time, util_mem, action (或 last_action), none_count, instance_count, usetime (或 use_time), timestamp
                                 注意：none_count 和 instance_count 共享特征分别包含 prompt/token 的两个值，两个代理都会同时包含这两个值
        :param debug_features: 是否输出归一化后的特征值和特征名称（用于调试）
        """
        assert mode in ("joint", "prompt", "token")
        
        self.cluster = cluster
        self.router = router
        self.applications = applications
        self.stack_size = stack_size
        self.mode = mode
        self.debug_features = debug_features  # 调试开关
        
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
                'kv_transferred_bytes': 0,
                'last_rps': 0.0  # 上一次的 RPS 值（用于计算 RPS 差值）
            }
        else:
            self.last_stats = RLStateCollector._shared_last_stats


    def _collect_snapshot(self, current_time, interval, last_prompt_action=None, last_token_action=None):
        """
        根据初始化时选择的特征配置，只收集需要的原始状态数据。
        使用类级别的缓存机制，确保在一次决策周期内只被调用一次。
        注意：为每个 mode 维护独立的缓存，避免 prompt 和 token collector 共享错误的缓存。
        
        :param last_prompt_action: 上一次 prompt 代理的动作值（可选）
        :param last_token_action: 上一次 token 代理的动作值（可选）
        """
        # 检查缓存：如果时间戳、间隔和动作值都相同，说明是同一个决策周期内的调用，直接返回缓存
        # 如果动作值为 None，则使用实例变量
        # 使用 mode 作为缓存键，确保不同 mode 的 collector 使用各自的缓存
        mode = self.mode
        cached_prompt_action = RLStateCollector._snapshot_cache_prompt_action.get(mode)
        cached_token_action = RLStateCollector._snapshot_cache_token_action.get(mode)
        current_prompt_action = last_prompt_action if last_prompt_action is not None else getattr(self, 'last_prompt_action', 0.0)
        current_token_action = last_token_action if last_token_action is not None else getattr(self, 'last_token_action', 0.0)
        
        if (mode in RLStateCollector._snapshot_cache and 
            RLStateCollector._snapshot_cache[mode] is not None and 
            RLStateCollector._snapshot_cache_time.get(mode) == current_time and
            RLStateCollector._snapshot_cache_interval.get(mode) == interval and
            cached_prompt_action == current_prompt_action and
            cached_token_action == current_token_action):
            return RLStateCollector._snapshot_cache[mode]
        
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
        
        # 2. RPS Delta (当前 RPS - 上一次 RPS)
        rps_delta = rps - self.last_stats['last_rps']
        if cfg.get("needs_rps_delta", False):
            snapshot.append(rps_delta)
        # 更新上一次的 RPS 值
        self.last_stats['last_rps'] = rps

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
        sch_p_queue_tokens, sch_d_queue_tokens, wait_time, avg_prompt_size, prompt_instance_pending_token = self.get_scheduler_feature()
        n_p, n_t, util_mem_p, util_mem_t, ins_p_queue, ins_d_queue = self.get_instance_feature()
        
        if cfg["needs_p_queue"]:
            snapshot.append(sch_p_queue_tokens)
        if cfg["needs_d_queue"]:
            snapshot.append(sch_d_queue_tokens)
        if cfg["needs_wait_time"]:
            snapshot.append(wait_time)
        if cfg.get("needs_prompt_instance_pending_token", False):
            snapshot.append(prompt_instance_pending_token)

        # --- C. 资源状态 (Resources) ---
        # 无论特征配置如何，都需要计算利用率用于奖励计算
        util_p, util_d = self.get_avg_utilization(current_time, interval)

        # 实例数量特征（类似 none_count，两个代理都同时包含 n_p 和 n_t）
        if cfg.get("needs_instance_count", False):
            snapshot.append(n_p)
            snapshot.append(n_t)
        if cfg["needs_util_p"]:
            snapshot.append(util_p)
        if cfg["needs_util_d"]:
            snapshot.append(util_d)
        if cfg["needs_util_mem"]:
            # 分别添加 prompt 和 token 实例的内存利用率
            snapshot.append(util_mem_p)
            snapshot.append(util_mem_t)

        # --- D. 性能反馈 (SLO) ---
        # 无论特征配置如何，都需要获取真实的SLO数据用于奖励计算
        # 根据 mode 传入参数：prompt 时返回并缓存结果，token 时取缓存结果并清除缓存
        ttft, tbt, vio_slo_rate, avg_queue_time, avg_nth_token_overhead, prompt_none_count, token_none_count = self.scheduler.get_period_result(mode=self.mode)
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
        # 类似 last_action，两个代理都同时包含 prompt_none_count 和 token_none_count
        if cfg.get("needs_none_count", False):
            snapshot.append(prompt_none_count)
            snapshot.append(token_none_count)

        # --- F1. 缩容特征 ---
        # 类似 none_count 和 instance_count，两个代理都同时包含 draining_prompt 和 draining_token
        if cfg.get("needs_draining", False):
            snapshot.append(draining_prompt)
            snapshot.append(draining_token)

        # --- G. 上一次动作特征 ---
        # 将上一次的动作添加到状态中，帮助 agent 了解自己之前做了什么决策
        # 根据配置决定是否包含
        # 两个代理都接收两个动作（prompt_action 和 token_action）
        if cfg.get("needs_last_action", False):
            # 如果提供了动作参数，使用它们；否则使用实例变量
            prompt_action = last_prompt_action if last_prompt_action is not None else getattr(self, 'last_prompt_action', 0.0)
            token_action = last_token_action if last_token_action is not None else getattr(self, 'last_token_action', 0.0)
            snapshot.append(prompt_action)
            snapshot.append(token_action)

        # --- H. 实例使用时间特征 ---
        # 根据配置决定是否包含
        use_time = self.get_usetime()

        if cfg.get("needs_usetime", False):
            snapshot.append(use_time)

        # --- I. 时间戳特征 ---
        # 根据配置决定是否包含
        # 使用归一化的时间戳（相对于最大时间，默认86400秒=1天）
        if cfg.get("needs_timestamp", False):
            # 归一化时间戳到 [0, 1] 范围
            # 使用 tanh 归一化，可以处理任意大小的时间戳
            MAX_TIME = 60.0  # 默认最大时间：1天（86400秒）
            normalized_timestamp = np.tanh(current_time / MAX_TIME)
            snapshot.append(normalized_timestamp)

        # --- 更新累积状态供下次使用 ---
        # 无论是否需要这些特征，都需要更新统计值，因为可能被其他 collector 使用
        self.last_stats.update({
            'arrival_count': curr_arrivals,
            'completed_tokens': curr_tokens,
            'completed_prompts': curr_prompts,
        })

        # reward_stats: 保持原有语义，给奖励函数使用的"快速指标"
        # 获取 usetime
        reward_stats = [prompt_rate, token_rate, sch_p_queue_tokens, sch_d_queue_tokens, n_p, n_t, avg_prompt_size, ttft, tbt,
                        ins_p_queue, ins_d_queue, avg_queue_time, avg_nth_token_overhead, use_time,rps]
        instance_num = [n_p, n_t, util_p, util_d]

        result = (np.array(snapshot, dtype=np.float32), instance_num, reward_stats, rps)
        
        # 缓存结果，供同一决策周期内的同一 mode 的 collector 使用
        # 使用 mode 作为缓存键，确保不同 mode 的 collector 使用各自的缓存
        mode = self.mode
        RLStateCollector._snapshot_cache[mode] = result
        RLStateCollector._snapshot_cache_time[mode] = current_time
        RLStateCollector._snapshot_cache_interval[mode] = interval
        # 缓存动作值，确保同一决策周期内使用相同的动作值
        prompt_action = last_prompt_action if last_prompt_action is not None else getattr(self, 'last_prompt_action', 0.0)
        token_action = last_token_action if last_token_action is not None else getattr(self, 'last_token_action', 0.0)
        RLStateCollector._snapshot_cache_prompt_action[mode] = prompt_action
        RLStateCollector._snapshot_cache_token_action[mode] = token_action

        return result

    # ------------------------- 归一化 ------------------------- #
    def _normalize(self, raw_vector):
        """
        根据模式对收集的原始特征进行归一化。
        由于 _collect_snapshot 已经按模式只收集需要的特征，这里直接按顺序归一化即可。
        """
        norm_vec = []
        feature_names = []  # 用于存储特征名称（如果开启调试）
        idx = 0
        cfg = self.feature_config
        MAX_INSTANCES = 100

        # Workload features
        if cfg["needs_rps"]:
            norm_vec.append(np.log1p(raw_vector[idx]+1) / 10.0)
            if self.debug_features:
                feature_names.append("rps")
            idx += 1
        if cfg.get("needs_rps_delta", False):
            # RPS 差值使用类似的归一化方式（log1p），保留符号
            rps_delta_val = raw_vector[idx]
            norm_rps_delta = np.sign(rps_delta_val) * np.log1p(np.abs(rps_delta_val) + 1) / 10.0
            norm_vec.append(norm_rps_delta)
            if self.debug_features:
                feature_names.append("rps_delta")
            idx += 1
        if cfg["needs_prompt_rate"]:
            norm_vec.append(np.log1p(raw_vector[idx]) / 20.0)
            if self.debug_features:
                feature_names.append("prompt_rate")
            idx += 1
        if cfg["needs_token_rate"]:
            norm_vec.append(np.log1p(raw_vector[idx]) / 20.0)
            if self.debug_features:
                feature_names.append("token_rate")
            idx += 1

        # Length features
        if cfg["needs_prompt_len"]:
            norm_vec.append(np.log1p(raw_vector[idx]+0.01) / 20.0)
            if self.debug_features:
                feature_names.append("prompt_len")
            idx += 1
        if cfg["needs_output_len"]:
            norm_vec.append(np.log1p(raw_vector[idx]+0.01) / 20.0)
            if self.debug_features:
                feature_names.append("output_len")
            idx += 1

        # Queue features
        if cfg["needs_p_queue"]:
            norm_vec.append(np.log1p(raw_vector[idx]+0.01) / 20.0)
            if self.debug_features:
                feature_names.append("p_queue")
            idx += 1
        if cfg["needs_d_queue"]:
            norm_vec.append(np.log1p(raw_vector[idx]+0.01) / 20.0)
            if self.debug_features:
                feature_names.append("d_queue")
            idx += 1
        if cfg["needs_wait_time"]:
            norm_vec.append(np.tanh(raw_vector[idx] / 10.0))
            if self.debug_features:
                feature_names.append("wait_time")
            idx += 1
        if cfg.get("needs_prompt_instance_pending_token", False):
            # 归一化 prompt_instance_pending_token，使用 log1p 归一化
            norm_vec.append(np.log1p(raw_vector[idx]+1) / 10.0)
            if self.debug_features:
                feature_names.append("prompt_instance_pending_token")
            idx += 1

        # Resource features
        # 实例数量特征（类似 none_count，两个代理都同时包含 n_p 和 n_t）
        if cfg.get("needs_instance_count", False):
            norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # n_p
            if self.debug_features:
                feature_names.append("n_p")
            idx += 1
            norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # n_t
            if self.debug_features:
                feature_names.append("n_t")
            idx += 1
        if cfg["needs_util_p"]:
            norm_vec.append(np.clip(raw_vector[idx], 0, 1))
            if self.debug_features:
                feature_names.append("util_p")
            idx += 1
        if cfg["needs_util_d"]:
            norm_vec.append(np.clip(raw_vector[idx], 0, 1))
            if self.debug_features:
                feature_names.append("util_d")
            idx += 1
        if cfg["needs_util_mem"]:
            # 分别归一化 prompt 和 token 实例的内存利用率
            norm_vec.append(np.clip(raw_vector[idx], 0, 1))  # util_mem_p
            if self.debug_features:
                feature_names.append("util_mem_p")
            idx += 1
            norm_vec.append(np.clip(raw_vector[idx], 0, 1))  # util_mem_t
            if self.debug_features:
                feature_names.append("util_mem_t")
            idx += 1

        # SLO features
        if cfg["needs_ttft"]:
            for i in range(3):
                norm_vec.append(np.tanh(raw_vector[idx]))
                if self.debug_features:
                    feature_names.append(f"ttft_p{['50', '90', '99'][i]}")
                idx += 1
        if cfg["needs_tbt"]:
            for i in range(3):
                norm_vec.append(np.tanh(raw_vector[idx]))
                if self.debug_features:
                    feature_names.append(f"tbt_p{['50', '90', '99'][i]}")
                idx += 1

        # Scaling features
        if cfg["needs_scaling_prompt"]:
            norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # scaling_up_prompt
            if self.debug_features:
                feature_names.append("scaling_up_prompt")
            idx += 1
            norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # draining_prompt
            if self.debug_features:
                feature_names.append("draining_prompt")
            idx += 1
        if cfg["needs_scaling_token"]:
            norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # scaling_up_token
            if self.debug_features:
                feature_names.append("scaling_up_token")
            idx += 1
            norm_vec.append(np.clip(raw_vector[idx] / MAX_INSTANCES, 0, 1))  # draining_token
            if self.debug_features:
                feature_names.append("draining_token")
            idx += 1

        # None count features (瓶颈指标，使用log1p归一化),最大值由请求量决定
        # 类似 last_action，两个代理都同时包含 prompt_none_count 和 token_none_count
        if cfg.get("needs_none_count", False):
            norm_prompt_none = (raw_vector[idx]+0.1) / 100.0
            norm_vec.append(norm_prompt_none)  # prompt_none_count
            if self.debug_features:
                feature_names.append("prompt_none_count")
            idx += 1
            norm_token_none = (raw_vector[idx]+0.1) / 100.0
            norm_vec.append(norm_token_none)  # token_none_count
            if self.debug_features:
                feature_names.append("token_none_count")
            idx += 1

        # Draining features (缩容特征，使用clip归一化)
        # 类似 none_count 和 instance_count，两个代理都同时包含 draining_prompt 和 draining_token
        if cfg.get("needs_draining", False):
            norm_vec.append(np.clip((raw_vector[idx]+0.1) / MAX_INSTANCES, 0, 1))  # draining_prompt
            if self.debug_features:
                feature_names.append("draining_prompt")
            idx += 1
            norm_vec.append(np.clip((raw_vector[idx]+0.1) / MAX_INSTANCES, 0, 1))  # draining_token
            if self.debug_features:
                feature_names.append("draining_token")
            idx += 1

        # Last action feature (动作通常在 [-1, 1] 范围内，使用 tanh 归一化到 [-1, 1])
        # 根据配置决定是否归一化
        # 两个代理都接收两个动作（prompt_action 和 token_action）
        if cfg.get("needs_last_action", False):
            norm_vec.append(np.tanh(raw_vector[idx]))  # last_prompt_action
            if self.debug_features:
                feature_names.append("last_prompt_action")
            idx += 1
            norm_vec.append(np.tanh(raw_vector[idx]))  # last_token_action
            if self.debug_features:
                feature_names.append("last_token_action")
            idx += 1

        # Use time feature (使用时间，使用 log1p 归一化)
        # 根据配置决定是否归一化
        if cfg.get("needs_usetime", False):
            # 使用 log1p 归一化，假设最大使用时间为 1000（可以根据实际情况调整）
            norm_vec.append(np.log1p(raw_vector[idx]) / np.log1p(200.0)) #最大实例数*决策间隔
            if self.debug_features:
                feature_names.append("use_time")
            idx += 1

        # Timestamp feature (时间戳特征，已经在 _collect_snapshot 中归一化)
        # 根据配置决定是否包含
        if cfg.get("needs_timestamp", False):
            # 时间戳已经在 _collect_snapshot 中使用 tanh 归一化，直接使用即可
            norm_vec.append(raw_vector[idx])
            if self.debug_features:
                feature_names.append("timestamp")
            idx += 1

        # 如果开启调试模式，输出特征名称和对应的归一化值（一行输出）
        if self.debug_features:
            feature_str = " | ".join([f"{name}={value:.4f}" for name, value in zip(feature_names, norm_vec)])
            logging.info(f"[{self.mode.upper()}] Features (dim={len(norm_vec)}): {feature_str}")

        return np.array(norm_vec, dtype=np.float32)

    def get_state_and_stats(self, current_time, interval, last_prompt_action=None, last_token_action=None):
        """
        统一接口，根据 mode 返回对应的堆叠状态（包含上一次动作）：
        - joint: (原维度 + 2) * stack_size（+2 是两个动作：prompt_action 和 token_action）
        - prompt: (原维度 + 2) * stack_size（+2 是两个动作：prompt_action 和 token_action）
        - token: (原维度 + 2) * stack_size（+2 是两个动作：prompt_action 和 token_action）
        
        :param last_prompt_action: 上一次 prompt 代理执行的动作值（可选），如果提供则更新
        :param last_token_action: 上一次 token 代理执行的动作值（可选），如果提供则更新
        """
        # 如果提供了上一次动作，更新它们
        if last_prompt_action is not None:
            self.last_prompt_action = float(last_prompt_action)
        if last_token_action is not None:
            self.last_token_action = float(last_token_action)
        
        raw_snapshot, instance_num, reward_stats, rps = self._collect_snapshot(
            current_time, interval, 
            last_prompt_action=last_prompt_action, 
            last_token_action=last_token_action
        )
        normalized = self._normalize(raw_snapshot)

        self.state_buffer.append(normalized)
        stacked_state = np.concatenate(self.state_buffer)

        return stacked_state, reward_stats, instance_num, rps

    @classmethod
    def clear_snapshot_cache(cls):
        """
        清除快照缓存，应该在每次决策周期结束时调用。
        确保下一个决策周期会重新收集快照。
        清除所有 mode 的缓存。
        """
        cls._snapshot_cache.clear()
        cls._snapshot_cache_time.clear()
        cls._snapshot_cache_interval.clear()
        cls._snapshot_cache_prompt_action.clear()
        cls._snapshot_cache_token_action.clear()

    def get_scheduler_feature(self):
        """
        获取调度器（scheduler）相关的特征
        
        Returns:
            tuple: (sch_p_queue_tokens, sch_d_queue_tokens, wait_time, avg_prompt_size, prompt_instance_pending_token)
                - sch_p_queue_tokens: 调度器中待处理的 prompt 队列长度（token 数）
                - sch_d_queue_tokens: 调度器中待处理的 token 队列长度（token 数）
                - wait_time: 平均等待时间
                - avg_prompt_size: 平均 prompt 大小
                - prompt_instance_pending_token: prompt 实例待处理的 token 归一化值
        """
        scheduler = self.scheduler
        total_pending_prompt_queue_length, total_pending_tokens, avg_time, avg_prompt_size, prompt_instance_pending_token\
            = scheduler.get_queue_stats()
        # print("sch堆积状态:",total_pending_prompt_queue_length,total_pending_tokens)

        return (
            total_pending_prompt_queue_length,
            total_pending_tokens,
            avg_time,
            avg_prompt_size,
            prompt_instance_pending_token
        )

    def get_instance_feature(self):
        """
        获取实例（instance）相关的特征
        
        Returns:
            tuple: (n_p, n_t, util_mem_p, util_mem_t, ins_p_queue, ins_d_queue)
                - n_p: 活跃的 prompt 实例数量
                - n_t: 活跃的 token 实例数量
                - util_mem_p: prompt 实例的平均内存利用率
                - util_mem_t: token 实例的平均内存利用率
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

        # -------- util_mem：分别计算 prompt 和 token 实例的内存利用率 --------
        if active_prompts:
            total_memory_p = sum(inst.memory for inst in active_prompts)/active_prompts[0].max_memory
            util_mem_p = total_memory_p / len(active_prompts)
        else:
            util_mem_p = 0.0

        if active_tokens:
            total_memory_t = sum(inst.memory for inst in active_tokens)/active_tokens[0].max_memory
            util_mem_t = total_memory_t / len(active_tokens)
        else:
            util_mem_t = 0.0

        return (
            n_p,
            n_t,
            util_mem_p,
            util_mem_t,
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

    def get_usetime(self):
        """
        获取实例使用时间（根据 mode 返回对应的 usetime）
        
        Returns:
            float: prompt 或 token 实例的使用时间
        """
        if not hasattr(self.applications[0], 'scaling_manager') or \
           self.applications[0].scaling_manager is None:
            return 0.0
        
        scaling_manager = self.applications[0].scaling_manager
        
        if self.mode == "prompt":
            return scaling_manager.calculate_prompt_instance_time_since_last()
        elif self.mode == "token":
            return scaling_manager.calculate_token_instance_time_since_last()
        else:  # joint mode
            # 对于 joint 模式，返回两者的总和
            return (scaling_manager.calculate_prompt_instance_time_since_last() +
                    scaling_manager.calculate_token_instance_time_since_last())

    @property
    def scheduler(self):
        return self.applications[0].scheduler

    @property
    def feature_dim(self):
        """
        根据模式返回不同的单步特征维度：
        - joint: 32 (rps + prompt_rate + token_rate + prompt_len + output_len + 
                     p_queue + d_queue + wait_time + prompt_instance_pending_token + 2*instance_count + util_p + util_d + 
                     2*util_mem (util_mem_p + util_mem_t) + 3*ttft_rate + 3*tbt_rate + 4*scaling + 
                     2*none_count + 2*draining + 2*last_action + use_time + timestamp)
        - prompt: 22 (rps + prompt_rate + prompt_len + p_queue + wait_time + prompt_instance_pending_token + 
                      2*instance_count + util_p + 2*util_mem (util_mem_p + util_mem_t) + 3*ttft_rate + 2*scaling_prompt + 
                      2*none_count + 2*draining + 2*last_action + use_time + timestamp)
        - token: 22 (rps + token_rate + output_len + d_queue + wait_time + prompt_instance_pending_token + 
                     2*instance_count + util_d + 2*util_mem (util_mem_p + util_mem_t) + 3*tbt_rate + 2*scaling_token + 
                     2*none_count + 2*draining + 2*last_action + use_time + timestamp)
        注意：none_count、instance_count、draining、last_action、use_time 和 timestamp 都是共享特征，两个代理都同时包含 prompt 和 token 的值
        注意：util_mem 现在包含两个值：util_mem_p（prompt实例内存利用率）和 util_mem_t（token实例内存利用率）
        注意：draining 包含两个值：draining_prompt（正在缩容的prompt实例数）和 draining_token（正在缩容的token实例数）
        注意：prompt_instance_pending_token 是 prompt 实例待处理的 token 归一化值
        """
        cfg = self.feature_config
        dim = 0
        
        # Workload
        if cfg["needs_rps"]:
            dim += 1
        if cfg.get("needs_rps_delta", False):
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
        if cfg.get("needs_prompt_instance_pending_token", False):
            dim += 1
        
        # Resource
        # 实例数量特征（类似 none_count，两个代理都同时包含两个值）
        if cfg.get("needs_instance_count", False):
            dim += 2  # n_p 和 n_t
        if cfg["needs_util_p"]:
            dim += 1
        if cfg["needs_util_d"]:
            dim += 1
        if cfg["needs_util_mem"]:
            dim += 2  # util_mem_p 和 util_mem_t
        
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
        
        # None count (瓶颈指标，类似 last_action，两个代理都同时包含两个值)
        if cfg.get("needs_none_count", False):
            dim += 2  # prompt_none_count 和 token_none_count

        # Draining (缩容特征，类似 none_count 和 instance_count，两个代理都同时包含两个值)
        if cfg.get("needs_draining", False):
            dim += 2  # draining_prompt 和 draining_token
        
        # Last action (根据配置决定是否包含)
        # 两个代理都接收两个动作（prompt_action 和 token_action）
        if cfg.get("needs_last_action", False):
            dim += 2  # prompt_action 和 token_action

        # Use time (根据配置决定是否包含)
        if cfg.get("needs_usetime", False):
            dim += 1  # use_time

        # Timestamp (根据配置决定是否包含)
        if cfg.get("needs_timestamp", False):
            dim += 1  # timestamp
        
        return dim