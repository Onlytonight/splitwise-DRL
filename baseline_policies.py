"""
非学习的基线扩缩容策略实现

支持的策略：
1. HeteroScale - 结合比例控制和延迟触发的生产级策略
2. UtilizationBased - 基于显存利用率的简单策略
3. QueueBased - 基于队列长度的简单策略
4. StaticPDRatio - 静态 P/D 实例数策略（保持固定实例数）
"""

import logging
import numpy as np
from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """基线策略的抽象基类"""
    
    def __init__(self, config):
        """
        初始化策略
        
        Args:
            config: 策略配置字典
        """
        self.config = config
        self.decision_count = 0
        
    @abstractmethod
    def decide(self, state_info, raw_stats, instance_info):
        """
        根据当前状态做出扩缩容决策
        
        Args:
            state_info: 状态信息字典，包含各种监控指标
            raw_stats: 原始统计数据 [prompt_rate, token_rate, p_queue, d_queue, ...]
            instance_info: 实例信息 [n_p, n_t, util_p, util_d, util_mem_p, util_mem_t]
            
        Returns:
            tuple: (delta_prompt, delta_token) - prompt/token池的实例数变化
        """
        pass
    
    def log_decision(self, decision_info):
        """记录决策信息"""
        self.decision_count += 1
        if self.decision_count % 10 == 0:
            logging.info(f"[{self.__class__.__name__}] Step {self.decision_count}: {decision_info}")


class HeteroScalePolicy(BasePolicy):
    """
    HeteroScale 策略
    
    结合比例控制和延迟触发两种机制：
    1. 比例控制：基于 TPS 的平稳扩缩容
    2. 延迟触发：基于 TBT 的紧急扩容
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # === 比例控制参数 ===
        # 目标单实例 Decode TPS（每秒生成的 token 数）
        self.target_decode_tps_per_instance = config.get("target_decode_tps_per_instance", 100)
        
        # P/D 比例（Prefill:Decode 的比例，例如 0.33 表示 1:3）
        self.pd_ratio = config.get("pd_ratio", 0.33)
        
        # 扩缩容阈值（变化率超过此值才触发）
        self.scale_out_threshold = config.get("scale_out_threshold", 0.1)  # 超过 1.1 倍扩容
        self.scale_in_threshold = config.get("scale_in_threshold", 0.1)    # 低于 0.9 倍缩容
        
        # === 延迟触发参数 ===
        # TBT SLO 阈值（毫秒）
        self.tbt_slo = config.get("tbt_slo", 50.0)  # 50ms
        # TBT 触发倍数（TBT > latency_panic_threshold × SLO 时紧急扩容）
        self.latency_panic_threshold = config.get("latency_panic_threshold", 1.2)
        # 紧急扩容倍数（直接按此倍数扩容，例如 1.2 表示扩容 20%）
        self.latency_panic_scale_factor = config.get("latency_panic_scale_factor", 1.2)
        # 是否启用延迟触发
        self.enable_latency_trigger = config.get("enable_latency_trigger", True)
        
        # === 实例数限制 ===
        self.min_instances_per_pool = config.get("min_instances_per_pool", 1)
        self.max_total_instances = config.get("max_total_instances", 100)
        
        logging.info(f"[HeteroScale] Initialized with target_decode_tps={self.target_decode_tps_per_instance}, "
                     f"pd_ratio={self.pd_ratio}, tbt_slo={self.tbt_slo}ms, "
                     f"latency_panic_threshold={self.latency_panic_threshold}")
    
    def decide(self, state_info, raw_stats, instance_info):
        """
        HeteroScale 决策算法
        
        包含两个机制：
        1. 比例控制（主推力）：基于 TPS 计算目标实例数
        2. 延迟触发（安全网）：基于 TBT 的紧急扩容
        
        公式：
        I_new_total = decode_tps_total / target_decode_tps_per_instance
        I_prefill = ⌈I_new_total / (1 + 1/pd_ratio)⌉
        I_decode = I_prefill × (1/pd_ratio)
        
        扩容条件: R > 1 + θ_out 或 TBT > latency_panic_threshold × SLO
        缩容条件: R < 1 - θ_in
        """
        # 解析输入数据
        # raw_stats: [prompt_rate, token_rate, p_queue, d_queue, n_p, n_t, avg_prompt_size, 
        #             ttft, tbt, ins_p_queue, ins_d_queue, avg_queue_time, avg_nth_token_overhead, use_time, rps]
        # 注意: ttft 和 tbt 是列表 [p50, p90, p99]
        prompt_rate, token_rate, p_queue, d_queue, n_p, n_t, avg_prompt_size, \
            ttft, tbt, ins_p_queue, ins_d_queue, avg_queue_time, avg_nth_token_overhead, use_time, rps = raw_stats[:15]
        
        # instance_info: [n_p, n_t, util_p, util_d, util_mem_p, util_mem_t]
        current_prompt_instances = instance_info[0]
        current_token_instances = instance_info[1]
        
        # decode_tps_total 是总的 TPS（所有 decode 实例的总和）
        # 在我们的系统中，token_rate 就是所有 decode 实例的总 TPS
        decode_tps_total = token_rate
        
        # 防止除零
        if current_token_instances == 0:
            current_token_instances = 1
        
        # ========================================
        # 1. 延迟触发检查（安全网，优先级最高）
        # ========================================
        if self.enable_latency_trigger:
            # 使用 TBT 的 P99 值来判断是否需要紧急扩容
            tbt_p99_slowdown = tbt[2] if isinstance(tbt, (list, tuple)) and len(tbt) >= 3 else tbt

            
            tbt_threshold = self.tbt_slo 

            if tbt_p99_slowdown > tbt_threshold and decode_tps_total > 0:
                # 紧急扩容：直接按百分比增加，不经过比例计算
                target_prompt = int(current_prompt_instances * self.latency_panic_scale_factor)
                target_token = int(current_token_instances * self.latency_panic_scale_factor)
                
                # 应用上下限
                target_prompt = max(0, min(target_prompt, self.max_total_instances))
                target_token = max(self.min_instances_per_pool, min(target_token, self.max_total_instances))
                
                delta_p = target_prompt - current_prompt_instances
                delta_t = target_token - current_token_instances
                
                self.log_decision({
                    "trigger": "latency_panic",
                    "tbt_p99": f"{tbt_p99_slowdown:.2f}",
                    "tbt_threshold": f"{tbt_threshold*1000:.2f}ms",
                    "delta_p": delta_p,
                    "delta_t": delta_t,
                    "reason": f"TBT P99 slowdown ({tbt_p99_slowdown:.2f}) > SLO ({self.tbt_slo} ), "
                             f"scale by {self.latency_panic_scale_factor}x"
                })
                
                return delta_p, delta_t
        
        # ========================================
        # 2. 比例控制（主推力）
        # ========================================
        

        
        # 计算目标总实例数（基于 TPS）
        # I_new_total = decode_tps_total / target_decode_tps_per_instance
        token_instances_needed = decode_tps_total / self.target_decode_tps_per_instance
        
        # 根据 P/D 比例分配
        # pd_ratio = P/D，例如 0.33 表示 1:3
        # 总份数 = 1 + (1/pd_ratio) = 1 + 3 = 4
        # Prefill 份数 = 1，Decode 份数 = 3
        if self.pd_ratio > 0:
            expected_token_instances = token_instances_needed
            expected_prompt_instances = expected_token_instances * self.pd_ratio
        else:
            # 如果 pd_ratio = 0，则只有 Decode 实例
            expected_prompt_instances = 0
            expected_token_instances = token_instances_needed
        
        # 计算变化率
        current_total = current_prompt_instances + current_token_instances
        expected_total = expected_prompt_instances + expected_token_instances
        
        if current_total > 0:
            ratio = expected_total / current_total
        else:
            ratio = 2.0  # 如果当前没有实例，直接扩容
        
        # === 决策逻辑 ===
        delta_p = 0
        delta_t = 0
        trigger = "no_change"
        
        # 扩容条件: R > 1 + θ_out
        if ratio > (1 + self.scale_out_threshold):
            target_prompt = int(round(expected_prompt_instances))
            target_token = int(round(expected_token_instances))
            
            # 应用上下限
            target_prompt = max(0, min(target_prompt, self.max_total_instances))
            target_token = max(self.min_instances_per_pool, min(target_token, self.max_total_instances))
            
            delta_p = target_prompt - current_prompt_instances
            delta_t = target_token - current_token_instances
            trigger = "scale_out"
            
            current_target_tps = current_token_instances * self.target_decode_tps_per_instance
            reason = f"decode_tps={decode_tps_total:.1f} > target={current_target_tps:.1f}"
        
        # 缩容条件: R < 1 - θ_in
        elif ratio < (1 - self.scale_in_threshold):
            target_prompt = int(round(expected_prompt_instances))
            target_token = int(round(expected_token_instances))
            
            # 应用上下限
            target_prompt = max(0, min(target_prompt, self.max_total_instances))
            target_token = max(self.min_instances_per_pool, min(target_token, self.max_total_instances))
            
            delta_p = target_prompt - current_prompt_instances
            delta_t = target_token - current_token_instances
            trigger = "scale_in"
            
            current_target_tps = current_token_instances * self.target_decode_tps_per_instance
            reason = f"decode_tps={decode_tps_total:.1f} < target={current_target_tps:.1f}"
        
        # 维持不变
        else:
            reason = "within threshold"
        
        # 记录决策信息
        self.log_decision({
            "trigger": trigger,
            "decode_tps_total": f"{decode_tps_total:.1f}",
            "current": f"P={current_prompt_instances} T={current_token_instances}",
            "expected": f"P={expected_prompt_instances:.2f} T={expected_token_instances:.2f}",
            "ratio": f"{ratio:.3f}",
            "delta_p": delta_p,
            "delta_t": delta_t,
            "reason": reason,
        })
        
        return delta_p, delta_t


class UtilizationBasedPolicy(BasePolicy):
    """
    基于显存利用率的扩缩容策略
    
    逻辑：
    - 显存利用率过高（> upper_threshold）-> 扩容
    - 显存利用率过低（< lower_threshold）-> 缩容
    - 分别监控 prompt 和 token 池
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 利用率阈值
        self.upper_threshold = config.get("upper_threshold", 0.8)  # 80%
        self.lower_threshold = config.get("lower_threshold", 0.3)  # 30%
        
        # 扩缩容步长
        self.scale_step = config.get("scale_step", 1)
        
        # 实例数限制
        self.min_instances_per_pool = config.get("min_instances_per_pool", 1)
        self.max_total_instances = config.get("max_total_instances", 100)
        
        logging.info(f"[UtilizationBased] Initialized with thresholds: "
                     f"[{self.lower_threshold}, {self.upper_threshold}], step={self.scale_step}")
    
    def decide(self, state_info, raw_stats, instance_info):
        """
        基于显存利用率的决策
        """
        # instance_info: [n_p, n_t, util_p, util_d, util_mem_p, util_mem_t]
        n_p, n_t, util_p, util_d, util_mem_p, util_mem_t = instance_info
        
        delta_p = 0
        delta_t = 0
        
        # === Prompt 池决策 ===
        if util_mem_p > self.upper_threshold:
            # 显存利用率过高，扩容
            delta_p = self.scale_step
            reason_p = f"mem_util={util_mem_p:.2f} > {self.upper_threshold}"
        elif util_mem_p < self.lower_threshold and n_p > self.min_instances_per_pool:
            # 显存利用率过低，缩容
            delta_p = -self.scale_step
            reason_p = f"mem_util={util_mem_p:.2f} < {self.lower_threshold}"
        else:
            reason_p = "maintain"
        
        # === Token 池决策 ===
        if util_mem_t > self.upper_threshold:
            # 显存利用率过高，扩容
            delta_t = self.scale_step
            reason_t = f"mem_util={util_mem_t:.2f} > {self.upper_threshold}"
        elif util_mem_t < self.lower_threshold and n_t > self.min_instances_per_pool:
            # 显存利用率过低，缩容
            delta_t = -self.scale_step
            reason_t = f"mem_util={util_mem_t:.2f} < {self.lower_threshold}"
        else:
            reason_t = "maintain"
        
        # 检查总实例数限制
        if delta_p > 0 or delta_t > 0:
            current_total = n_p + n_t
            new_total = current_total + delta_p + delta_t
            if new_total > self.max_total_instances:
                # 超出限制，不扩容
                delta_p = 0
                delta_t = 0
                reason_p = "max_limit"
                reason_t = "max_limit"
        
        # 记录决策信息
        self.log_decision({
            "n_p": n_p,
            "n_t": n_t,
            "util_mem_p": f"{util_mem_p:.2f}",
            "util_mem_t": f"{util_mem_t:.2f}",
            "delta_p": delta_p,
            "delta_t": delta_t,
            "reason_p": reason_p,
            "reason_t": reason_t,
        })
        
        return delta_p, delta_t


class QueueBasedPolicy(BasePolicy):
    """
    基于队列长度的扩缩容策略
    
    逻辑：
    - 队列过长（> upper_threshold）-> 扩容
    - 队列过短（< lower_threshold）-> 缩容
    - 分别监控 prompt 和 token 队列
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 队列长度阈值（token 数）
        self.prompt_queue_upper = config.get("prompt_queue_upper", 1000)
        self.prompt_queue_lower = config.get("prompt_queue_lower", 100)
        self.token_queue_upper = config.get("token_queue_upper", 5000)
        self.token_queue_lower = config.get("token_queue_lower", 500)
        
        # 扩缩容步长
        self.scale_step = config.get("scale_step", 1)
        
        # 实例数限制
        self.min_instances_per_pool = config.get("min_instances_per_pool", 1)
        self.max_total_instances = config.get("max_total_instances", 100)
        
        logging.info(f"[QueueBased] Initialized with thresholds: "
                     f"prompt=[{self.prompt_queue_lower}, {self.prompt_queue_upper}], "
                     f"token=[{self.token_queue_lower}, {self.token_queue_upper}]")
    
    def decide(self, state_info, raw_stats, instance_info):
        """
        基于队列长度的决策
        """
        # raw_stats: [prompt_rate, token_rate, p_queue, d_queue, ...]
        prompt_rate, token_rate, p_queue, d_queue = raw_stats[:4]
        n_p, n_t = instance_info[:2]
        
        delta_p = 0
        delta_t = 0
        
        # === Prompt 队列决策 ===
        if p_queue > self.prompt_queue_upper:
            # 队列过长，扩容
            delta_p = self.scale_step
            reason_p = f"queue={p_queue:.0f} > {self.prompt_queue_upper}"
        elif p_queue < self.prompt_queue_lower and n_p > self.min_instances_per_pool:
            # 队列过短，缩容
            delta_p = -self.scale_step
            reason_p = f"queue={p_queue:.0f} < {self.prompt_queue_lower}"
        else:
            reason_p = "maintain"
        
        # === Token 队列决策 ===
        if d_queue > self.token_queue_upper:
            # 队列过长，扩容
            delta_t = self.scale_step
            reason_t = f"queue={d_queue:.0f} > {self.token_queue_upper}"
        elif d_queue < self.token_queue_lower and n_t > self.min_instances_per_pool:
            # 队列过短，缩容
            delta_t = -self.scale_step
            reason_t = f"queue={d_queue:.0f} < {self.token_queue_lower}"
        else:
            reason_t = "maintain"
        
        # 检查总实例数限制
        if delta_p > 0 or delta_t > 0:
            current_total = n_p + n_t
            new_total = current_total + delta_p + delta_t
            if new_total > self.max_total_instances:
                delta_p = 0
                delta_t = 0
                reason_p = "max_limit"
                reason_t = "max_limit"
        
        # 记录决策信息
        self.log_decision({
            "p_queue": f"{p_queue:.0f}",
            "d_queue": f"{d_queue:.0f}",
            "n_p": n_p,
            "n_t": n_t,
            "delta_p": delta_p,
            "delta_t": delta_t,
            "reason_p": reason_p,
            "reason_t": reason_t,
        })
        
        return delta_p, delta_t


class StaticPDRatioPolicy(BasePolicy):
    """
    静态 P/D 实例数策略
    
    逻辑：
    - 保持固定的 prompt 和 token 实例数
    - 不根据负载动态调整
    - 只在初始化时设置实例数，之后保持不变
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 固定的实例数配置
        self.fixed_prompt_instances = config.get("fixed_prompt_instances", 1)
        self.fixed_token_instances = config.get("fixed_token_instances", 3)
        
        # 实例数限制（用于验证配置是否合理）
        self.min_instances_per_pool = config.get("min_instances_per_pool", 1)
        self.max_total_instances = config.get("max_total_instances", 100)
        
        # 验证配置
        if self.fixed_prompt_instances < self.min_instances_per_pool:
            logging.warning(f"[StaticPDRatio] fixed_prompt_instances ({self.fixed_prompt_instances}) "
                          f"< min_instances_per_pool ({self.min_instances_per_pool}), "
                          f"using min value")
            self.fixed_prompt_instances = self.min_instances_per_pool
        
        if self.fixed_token_instances < self.min_instances_per_pool:
            logging.warning(f"[StaticPDRatio] fixed_token_instances ({self.fixed_token_instances}) "
                          f"< min_instances_per_pool ({self.min_instances_per_pool}), "
                          f"using min value")
            self.fixed_token_instances = self.min_instances_per_pool
        
        total_instances = self.fixed_prompt_instances + self.fixed_token_instances
        if total_instances > self.max_total_instances:
            logging.warning(f"[StaticPDRatio] Total instances ({total_instances}) "
                          f"> max_total_instances ({self.max_total_instances}), "
                          f"scaling down proportionally")
            scale_factor = self.max_total_instances / total_instances
            self.fixed_prompt_instances = max(self.min_instances_per_pool, 
                                             int(self.fixed_prompt_instances * scale_factor))
            self.fixed_token_instances = max(self.min_instances_per_pool,
                                           int(self.fixed_token_instances * scale_factor))
        
        # 标记是否已经初始化过（第一次决策时设置实例数）
        self.initialized = False
        
        logging.info(f"[StaticPDRatio] Initialized with fixed instances: "
                    f"prompt={self.fixed_prompt_instances}, token={self.fixed_token_instances}")
    
    def decide(self, state_info, raw_stats, instance_info):
        """
        静态 P/D 实例数决策
        
        第一次调用时，将实例数调整到目标值
        之后保持不变
        """
        # instance_info: [n_p, n_t, util_p, util_d, util_mem_p, util_mem_t]
        current_prompt_instances = instance_info[0]
        current_token_instances = instance_info[1]
        
        # 第一次决策时，调整到目标实例数
        if not self.initialized:
            delta_p = self.fixed_prompt_instances - current_prompt_instances
            delta_t = self.fixed_token_instances - current_token_instances
            
            self.initialized = True
            
            self.log_decision({
                "trigger": "initial_setup",
                "current": f"P={current_prompt_instances} T={current_token_instances}",
                "target": f"P={self.fixed_prompt_instances} T={self.fixed_token_instances}",
                "delta_p": delta_p,
                "delta_t": delta_t,
                "reason": "Initial setup to fixed instances"
            })
            
            return delta_p, delta_t
        
        # 之后保持不变，但检查是否需要调整（如果实例数被外部改变）
        delta_p = self.fixed_prompt_instances - current_prompt_instances
        delta_t = self.fixed_token_instances - current_token_instances
        
        # 如果实例数已经正确，返回 0
        if delta_p == 0 and delta_t == 0:
            return 0, 0
        
        # 否则调整回目标值
        self.log_decision({
            "trigger": "maintain",
            "current": f"P={current_prompt_instances} T={current_token_instances}",
            "target": f"P={self.fixed_prompt_instances} T={self.fixed_token_instances}",
            "delta_p": delta_p,
            "delta_t": delta_t,
            "reason": "Maintaining fixed instances"
        })
        
        return delta_p, delta_t


# 策略工厂函数
def create_policy(policy_name, config):
    """
    创建策略实例
    
    Args:
        policy_name: 策略名称 ("heteroscale" / "utilization" / "queue" / "static_pd")
        config: 策略配置
        
    Returns:
        BasePolicy 实例
    """
    policy_map = {
        "heteroscale": HeteroScalePolicy,
        "utilization": UtilizationBasedPolicy,
        "queue": QueueBasedPolicy,
        "static_pd": StaticPDRatioPolicy,
    }
    
    policy_class = policy_map.get(policy_name.lower())
    if policy_class is None:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(policy_map.keys())}")
    
    return policy_class(config)
