"""
Autoscaling Policies for LLM Inference Serving.

This module provides various autoscaling strategies including:
- HeteroScale (TPS-based)
- Utilization-based (GPU Util)
- Latency-based (TTFT/TBT)
- Periodic/Cron-based
- RL-based (placeholder for reinforcement learning)
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from enum import Enum
from simulator import clock


class ScalingDecision(Enum):
    """扩缩容决策类型"""
    SCALE_OUT = "scale_out"  # 扩容
    SCALE_IN = "scale_in"    # 缩容
    NO_CHANGE = "no_change"  # 维持不变


class ScalingAction:
    """扩缩容动作"""
    def __init__(self, 
                 decision: ScalingDecision,
                 target_prompt_instances: int = 0,
                 target_token_instances: int = 0,
                 reason: str = ""):
        self.decision = decision
        self.target_prompt_instances = target_prompt_instances
        self.target_token_instances = target_token_instances
        self.reason = reason
    
    def __repr__(self):
        return (f"ScalingAction(decision={self.decision.value}, "
                f"prompt={self.target_prompt_instances}, "
                f"token={self.target_token_instances}, "
                f"reason='{self.reason}')")


class AutoscalingPolicy(ABC):
    """
    自动扩缩容策略的抽象基类
    
    所有扩缩容策略（包括基线和强化学习）都应继承此类
    """
    
    def __init__(self, 
                 name: str,
                 scale_out_cooldown: float = 180.0,  # 扩容冷却时间（秒）
                 scale_in_cooldown: float = 600.0,   # 缩容冷却时间（秒）
                 debug: bool = False):
        """
        初始化扩缩容策略
        
        Args:
            name: 策略名称
            scale_out_cooldown: 扩容冷却时间（秒）
            scale_in_cooldown: 缩容冷却时间（秒）
            debug: 是否启用调试日志
        """
        self.name = name
        self.scale_out_cooldown = scale_out_cooldown
        self.scale_in_cooldown = scale_in_cooldown
        self.debug = debug
        
        # 跟踪上次扩缩容时间
        self.last_scale_out_time = -float('inf')
        self.last_scale_in_time = -float('inf')
        
        # 日志
        self.logger = logging.getLogger(f"autoscaling.{name}")
        if debug:
            self.logger.setLevel(logging.DEBUG)
    
    @abstractmethod
    def decide(self, metrics: Dict) -> ScalingAction:
        """
        根据当前指标决定扩缩容动作
        
        Args:
            metrics: 包含各种系统指标的字典，例如：
                - 'current_prompt_instances': 当前 Prefill 实例数
                - 'current_token_instances': 当前 Decode 实例数
                - 'decode_tps': Decode TPS
                - 'prefill_gpu_util': Prefill GPU 利用率
                - 'decode_gpu_util': Decode GPU 利用率
                - 'ttft': Time to First Token (首字延迟)
                - 'tbt': Time Between Tokens (字间延迟)
                - 'timestamp': 当前时间戳
                - ...
        
        Returns:
            ScalingAction: 扩缩容决策
        """
        pass
    
    def can_scale_out(self) -> bool:
        """检查是否可以执行扩容（冷却时间）"""
        return (clock() - self.last_scale_out_time) >= self.scale_out_cooldown
    
    def can_scale_in(self) -> bool:
        """检查是否可以执行缩容（冷却时间）"""
        return (clock() - self.last_scale_in_time) >= self.scale_in_cooldown
    
    def record_scale_out(self):
        """记录扩容时间"""
        self.last_scale_out_time = clock()
    
    def record_scale_in(self):
        """记录缩容时间"""
        self.last_scale_in_time = clock()
    
    def reset(self):
        """重置策略状态（用于新的实验）"""
        self.last_scale_out_time = -float('inf')
        self.last_scale_in_time = -float('inf')


class HeteroScalePolicy(AutoscalingPolicy):
    """
    HeteroScale 策略 (TPS-based + Latency-based)
    
    核心机制：
    1. 比例控制（主推力）：基于 Decode TPS 计算期望实例数
       - I_new_total = I_curr × (TPS_curr / TPS_target)
       - 维持固定的 P/D 比例
    
    2. 延迟触发（安全网）：基于 TBT 的负反馈控制
       - 当 TBT > 1.2 × SLO 时，直接按百分比扩容（1.2倍）
       - 快速响应突发流量
    
    参考：HeteroScale 论文 Algorithm 2
    """
    
    def __init__(self,
                 target_decode_tps_per_instance: float = 100.0,  # 单实例目标 Decode TPS
                 pd_ratio: float = 0.33,  # P/D 比例 (Prefill / Decode)，0.33 表示 1:3
                 scale_out_threshold: float = 0.1,  # 扩容阈值 (10%)
                 scale_in_threshold: float = 0.1,   # 缩容阈值 (10%)
                 min_instances: int = 1,            # 最小实例数
                 max_instances: int = 100,          # 最大实例数
                 # 延迟触发参数
                 enable_latency_trigger: bool = True,  # 是否启用延迟触发
                 tbt_slo: float = 0.04,              # TBT SLO (秒)
                 latency_panic_threshold: float = 1.2,  # 延迟恐慌阈值 (1.2x)
                 latency_panic_scale_factor: float = 1.2,  # 恐慌时扩容倍数
                 **kwargs):
        super().__init__(name="HeteroScale", **kwargs)
        
        self.target_decode_tps_per_instance = target_decode_tps_per_instance
        self.pd_ratio = pd_ratio
        self.scale_out_threshold = scale_out_threshold
        self.scale_in_threshold = scale_in_threshold
        self.min_instances = min_instances
        self.max_instances = max_instances
        
        # 延迟触发参数
        self.enable_latency_trigger = enable_latency_trigger
        self.tbt_slo = tbt_slo
        self.latency_panic_threshold = latency_panic_threshold
        self.latency_panic_scale_factor = latency_panic_scale_factor
    
    def decide(self, metrics: Dict) -> ScalingAction:
        """
        HeteroScale 决策算法
        
        包含两个机制：
        1. 比例控制（主推力）：基于 TPS 计算目标实例数
        2. 延迟触发（安全网）：基于 TBT 的紧急扩容
        
        公式：
        I_new_total = I_curr × (TPS_curr / TPS_target)
        I_prefill = ⌈I_new_total / (1 + 1/pd_ratio)⌉
        I_decode = I_prefill × (1/pd_ratio)
        
        扩容条件: R > 1 + θ_out 或 TBT > 1.2 × SLO
        缩容条件: R < 1 - θ_in
        """
        current_token_instances = metrics.get('current_token_instances', 1)
        current_prompt_instances = metrics.get('current_prompt_instances', 0)
        decode_tps_total = metrics.get('decode_tps', 0.0)  # 总的 TPS（所有 decode 实例）
        tbt = metrics.get('tbt', 0.0)  # Time Between Tokens
        
        # 防止除零
        if current_token_instances == 0:
            current_token_instances = 1
        
        # === 1. 延迟触发检查（安全网，优先级最高）===
        if self.enable_latency_trigger and tbt > 0:
            tbt_threshold = self.tbt_slo * self.latency_panic_threshold
            if tbt > tbt_threshold and self.can_scale_out():
                # 紧急扩容：直接按百分比增加，不经过比例计算
                target_prompt = int(current_prompt_instances * self.latency_panic_scale_factor)
                target_token = int(current_token_instances * self.latency_panic_scale_factor)
                
                # 应用上下限
                target_prompt = max(self.min_instances, min(target_prompt, self.max_instances))
                target_token = max(self.min_instances, min(target_token, self.max_instances))
                
                self.record_scale_out()
                
                reason = (f"LATENCY_PANIC: tbt={tbt:.3f}s > {tbt_threshold:.3f}s "
                         f"(scale by {self.latency_panic_scale_factor}x)")
                
                if self.debug:
                    self.logger.debug(
                        f"[HeteroScale] {reason}, "
                        f"prompt: {current_prompt_instances}->{target_prompt}, "
                        f"token: {current_token_instances}->{target_token}"
                    )
                
                return ScalingAction(
                    decision=ScalingDecision.SCALE_OUT,
                    target_prompt_instances=target_prompt,
                    target_token_instances=target_token,
                    reason=reason
                )
        
        # === 2. 比例控制（主推力）===
        # 计算目标总实例数（基于 TPS）
        # I_new_total = TPS_curr / TPS_target_per_instance
        total_instances_needed = decode_tps_total / self.target_decode_tps_per_instance
        
        # 根据 P/D 比例分配
        # pd_ratio = P/D，例如 0.33 表示 1:3
        # 总份数 = 1 + (1/pd_ratio) = 1 + 3 = 4
        # Prefill 份数 = 1，Decode 份数 = 3
        if self.pd_ratio > 0:
            total_parts = 1 + (1 / self.pd_ratio)  # 例如：1 + 3 = 4
            expected_prompt_instances = total_instances_needed / total_parts
            expected_token_instances = expected_prompt_instances * (1 / self.pd_ratio)
        else:
            # 如果 pd_ratio = 0，则只有 Decode 实例
            expected_prompt_instances = 0
            expected_token_instances = total_instances_needed
        
        # 计算变化率
        current_total = current_prompt_instances + current_token_instances
        expected_total = expected_prompt_instances + expected_token_instances
        
        if current_total > 0:
            ratio = expected_total / current_total
        else:
            ratio = 2.0  # 如果当前没有实例，直接扩容
        
        if self.debug:
            self.logger.debug(
                f"[HeteroScale] decode_tps={decode_tps_total:.2f}, "
                f"current: P={current_prompt_instances} T={current_token_instances}, "
                f"expected: P={expected_prompt_instances:.2f} T={expected_token_instances:.2f}, "
                f"ratio={ratio:.3f}"
            )
        
        # === 决策逻辑 ===
        # 扩容条件: R > 1 + θ_out
        if ratio > (1 + self.scale_out_threshold) and self.can_scale_out():
            target_prompt = int(round(expected_prompt_instances))
            target_token = int(round(expected_token_instances))
            
            # 应用上下限
            target_prompt = max(0, min(target_prompt, self.max_instances))
            target_token = max(self.min_instances, min(target_token, self.max_instances))
            
            self.record_scale_out()
            
            current_target_tps = current_token_instances * self.target_decode_tps_per_instance
            reason = (f"decode_tps={decode_tps_total:.1f} > target={current_target_tps:.1f}")
            
            return ScalingAction(
                decision=ScalingDecision.SCALE_OUT,
                target_prompt_instances=target_prompt,
                target_token_instances=target_token,
                reason=reason
            )
        
        # 缩容条件: R < 1 - θ_in
        elif ratio < (1 - self.scale_in_threshold) and self.can_scale_in():
            target_prompt = int(round(expected_prompt_instances))
            target_token = int(round(expected_token_instances))
            
            # 应用上下限
            target_prompt = max(0, min(target_prompt, self.max_instances))
            target_token = max(self.min_instances, min(target_token, self.max_instances))
            
            self.record_scale_in()
            
            current_target_tps = current_token_instances * self.target_decode_tps_per_instance
            reason = (f"decode_tps={decode_tps_total:.1f} < target={current_target_tps:.1f}")
            
            return ScalingAction(
                decision=ScalingDecision.SCALE_IN,
                target_prompt_instances=target_prompt,
                target_token_instances=target_token,
                reason=reason
            )
        
        # 维持不变
        else:
            return ScalingAction(
                decision=ScalingDecision.NO_CHANGE,
                target_prompt_instances=current_prompt_instances,
                target_token_instances=current_token_instances,
                reason="within threshold"
            )


class HPAGPUPolicy(AutoscalingPolicy):
    """
    Baseline 1: HPA-GPU (基于硬件利用率的传统方式)
    
    这是最常见的 Kubernetes HPA 默认方式
    
    特点：
    - Prefill 池和 Decode 池完全独立，互不通信
    - 各自监控自己的 GPU 利用率
    - 采用比例控制公式：I_target = ⌈I_curr × (Util_observed / Util_target)⌉
    
    预期失败点：
    - Decode 池即使在没流量时也不缩容
    - 因为 Decode GPU 利用率受 KV Cache 显存管理影响
    - 即使负载低，利用率也常年维持在 80%-90% (Memory-bound)
    """
    
    def __init__(self,
                 target_prefill_util: float = 0.7,   # Prefill 目标利用率 (70%)
                 target_decode_util: float = 0.7,    # Decode 目标利用率 (70%)
                 scale_out_threshold: float = 0.1,   # 扩容阈值
                 scale_in_threshold: float = 0.1,    # 缩容阈值
                 min_instances: int = 1,
                 max_instances: int = 100,
                 **kwargs):
        super().__init__(name="HPA-GPU", **kwargs)
        
        self.target_prefill_util = target_prefill_util
        self.target_decode_util = target_decode_util
        self.scale_out_threshold = scale_out_threshold
        self.scale_in_threshold = scale_in_threshold
        self.min_instances = min_instances
        self.max_instances = max_instances
    
    def decide(self, metrics: Dict) -> ScalingAction:
        """
        HPA-GPU 决策算法：Prefill 池和 Decode 池独立扩缩容
        
        公式：I_target = ⌈I_curr × (Util_observed / Util_target)⌉
        
        特性：完全独立，互不通信
        """
        # 获取当前实例数和利用率
        current_prompt_instances = metrics.get('current_prompt_instances', 0)
        current_token_instances = metrics.get('current_token_instances', 1)
        prefill_util = metrics.get('prefill_gpu_util', 0.0)
        decode_util = metrics.get('decode_gpu_util', 0.0)
        
        # 防止除零
        if current_prompt_instances == 0:
            current_prompt_instances = 1
        if current_token_instances == 0:
            current_token_instances = 1
        
        # === Prefill 池独立决策 ===
        expected_prompt = current_prompt_instances * (prefill_util / self.target_prefill_util)
        prompt_ratio = expected_prompt / current_prompt_instances
        
        # === Decode 池独立决策 ===
        expected_token = current_token_instances * (decode_util / self.target_decode_util)
        token_ratio = expected_token / current_token_instances
        
        if self.debug:
            self.logger.debug(
                f"[HPA-GPU] Prefill: util={prefill_util:.2f}, current={current_prompt_instances}, "
                f"expected={expected_prompt:.2f}, ratio={prompt_ratio:.3f} | "
                f"Decode: util={decode_util:.2f}, current={current_token_instances}, "
                f"expected={expected_token:.2f}, ratio={token_ratio:.3f}"
            )
        
        # 判断是否需要扩缩容
        need_scale_out = (prompt_ratio > (1 + self.scale_out_threshold) or 
                         token_ratio > (1 + self.scale_out_threshold))
        need_scale_in = (prompt_ratio < (1 - self.scale_in_threshold) and 
                        token_ratio < (1 - self.scale_in_threshold))
        
        # 计算目标实例数
        target_prompt = int(round(expected_prompt))
        target_token = int(round(expected_token))
        
        # 应用上下限
        target_prompt = max(0, min(target_prompt, self.max_instances))
        target_token = max(self.min_instances, min(target_token, self.max_instances))
        
        # 扩容
        if need_scale_out and self.can_scale_out():
            self.record_scale_out()
            reason = (f"Prefill: util={prefill_util:.2f}/{self.target_prefill_util:.2f}, "
                     f"Decode: util={decode_util:.2f}/{self.target_decode_util:.2f}")
            
            return ScalingAction(
                decision=ScalingDecision.SCALE_OUT,
                target_prompt_instances=target_prompt,
                target_token_instances=target_token,
                reason=reason
            )
        
        # 缩容
        elif need_scale_in and self.can_scale_in():
            self.record_scale_in()
            reason = (f"Prefill: util={prefill_util:.2f}/{self.target_prefill_util:.2f}, "
                     f"Decode: util={decode_util:.2f}/{self.target_decode_util:.2f}")
            
            return ScalingAction(
                decision=ScalingDecision.SCALE_IN,
                target_prompt_instances=target_prompt,
                target_token_instances=target_token,
                reason=reason
            )
        
        # 维持不变
        else:
            return ScalingAction(
                decision=ScalingDecision.NO_CHANGE,
                target_prompt_instances=current_prompt_instances,
                target_token_instances=current_token_instances,
                reason="within threshold"
            )


class IndependentTPSPolicy(AutoscalingPolicy):
    """
    Baseline 2: Independent Scaling (基于 TPS 但不协同)
    
    特点：
    - Prefill 池监控自己的 Prefill TPS
    - Decode 池监控自己的 Decode TPS
    - 两个池子完全独立，互不通信
    
    公式：
    - I_p_target = I_p_curr × (Prefill_TPS / P_TPS_Target)
    - I_d_target = I_d_curr × (Decode_TPS / D_TPS_Target)
    
    预期失败点：
    - P/D 比例剧烈抖动
    - 如果 P 池扩容快而 D 池扩容慢，P 池处理了大量 Prompt 塞给 D 池
    - D 池接不住，导致 TBT 飙升
    """
    
    def __init__(self,
                 target_prefill_tps: float = 30.0,   # 单 Prefill 实例目标 TPS
                 target_decode_tps: float = 100.0,   # 单 Decode 实例目标 TPS
                 scale_out_threshold: float = 0.1,   # 扩容阈值
                 scale_in_threshold: float = 0.1,    # 缩容阈值
                 min_instances: int = 1,
                 max_instances: int = 100,
                 **kwargs):
        super().__init__(name="IndependentTPS", **kwargs)
        
        self.target_prefill_tps = target_prefill_tps
        self.target_decode_tps = target_decode_tps
        self.scale_out_threshold = scale_out_threshold
        self.scale_in_threshold = scale_in_threshold
        self.min_instances = min_instances
        self.max_instances = max_instances
    
    def decide(self, metrics: Dict) -> ScalingAction:
        """
        Independent TPS 决策算法：P 池只管自己的 TPS，D 池只管自己的 TPS
        
        公式：
        - I_p_target = I_p_curr × (Prefill_TPS / P_TPS_Target)
        - I_d_target = I_d_curr × (Decode_TPS / D_TPS_Target)
        """
        # 获取当前实例数和 TPS
        current_prompt_instances = metrics.get('current_prompt_instances', 0)
        current_token_instances = metrics.get('current_token_instances', 1)
        prefill_tps = metrics.get('prefill_tps', 0.0)  # Prefill TPS
        decode_tps = metrics.get('decode_tps', 0.0)    # Decode TPS
        
        # 防止除零
        if current_prompt_instances == 0:
            current_prompt_instances = 1
        if current_token_instances == 0:
            current_token_instances = 1
        
        # === Prefill 池独立决策 ===
        # Prefill TPS 是总的，需要计算单实例 TPS
        prefill_tps_per_instance = prefill_tps / current_prompt_instances if prefill_tps > 0 else 0
        expected_prompt = current_prompt_instances * (prefill_tps_per_instance / self.target_prefill_tps) if self.target_prefill_tps > 0 else current_prompt_instances
        prompt_ratio = expected_prompt / current_prompt_instances
        
        # === Decode 池独立决策 ===
        # Decode TPS 是总的，需要计算单实例 TPS
        decode_tps_per_instance = decode_tps / current_token_instances if decode_tps > 0 else 0
        expected_token = current_token_instances * (decode_tps_per_instance / self.target_decode_tps) if self.target_decode_tps > 0 else current_token_instances
        token_ratio = expected_token / current_token_instances
        
        if self.debug:
            self.logger.debug(
                f"[IndependentTPS] Prefill: tps={prefill_tps:.2f}, per_inst={prefill_tps_per_instance:.2f}, "
                f"current={current_prompt_instances}, expected={expected_prompt:.2f}, ratio={prompt_ratio:.3f} | "
                f"Decode: tps={decode_tps:.2f}, per_inst={decode_tps_per_instance:.2f}, "
                f"current={current_token_instances}, expected={expected_token:.2f}, ratio={token_ratio:.3f}"
            )
        
        # 判断是否需要扩缩容
        need_scale_out = (prompt_ratio > (1 + self.scale_out_threshold) or 
                         token_ratio > (1 + self.scale_out_threshold))
        need_scale_in = (prompt_ratio < (1 - self.scale_in_threshold) and 
                        token_ratio < (1 - self.scale_in_threshold))
        
        # 计算目标实例数
        target_prompt = int(round(expected_prompt))
        target_token = int(round(expected_token))
        
        # 应用上下限
        target_prompt = max(0, min(target_prompt, self.max_instances))
        target_token = max(self.min_instances, min(target_token, self.max_instances))
        
        # 扩容
        if need_scale_out and self.can_scale_out():
            self.record_scale_out()
            reason = (f"Prefill: tps={prefill_tps:.1f} (target={self.target_prefill_tps * current_prompt_instances:.1f}), "
                     f"Decode: tps={decode_tps:.1f} (target={self.target_decode_tps * current_token_instances:.1f})")
            
            return ScalingAction(
                decision=ScalingDecision.SCALE_OUT,
                target_prompt_instances=target_prompt,
                target_token_instances=target_token,
                reason=reason
            )
        
        # 缩容
        elif need_scale_in and self.can_scale_in():
            self.record_scale_in()
            reason = (f"Prefill: tps={prefill_tps:.1f} (target={self.target_prefill_tps * current_prompt_instances:.1f}), "
                     f"Decode: tps={decode_tps:.1f} (target={self.target_decode_tps * current_token_instances:.1f})")
            
            return ScalingAction(
                decision=ScalingDecision.SCALE_IN,
                target_prompt_instances=target_prompt,
                target_token_instances=target_token,
                reason=reason
            )
        
        # 维持不变
        else:
            return ScalingAction(
                decision=ScalingDecision.NO_CHANGE,
                target_prompt_instances=current_prompt_instances,
                target_token_instances=current_token_instances,
                reason="within threshold"
            )


class PureLatencyPolicy(AutoscalingPolicy):
    """
    Baseline 3: Pure Latency Scaling (纯延迟驱动/负反馈)
    
    特点：
    - 不参考吞吐量，只看用户卡不卡的反应式方法
    - 监控 P90 TBT (Time Between Tokens)
    - 使用步进式扩缩容（每次加/减固定数量的 Pod）
    
    扩缩逻辑：
    - If TBT > SLO × 1.1: Count = Count + Step (例如每次加 2 个 Pod)
    - If TBT < SLO × 0.8: Count = Count - Step (例如每次减 1 个 Pod)
    
    预期失败点：
    - 严重的扩缩容振荡（Flapping）
    - 延迟和 Pod 数量是非线性关系
    - 加了 2 个 Pod 后，延迟可能瞬间降到极低，触发缩容
    - 缩容后延迟又瞬间飙升，导致系统不停地在"增-删-增"之间跳变
    """
    
    def __init__(self,
                 tbt_slo: float = 0.04,             # TBT SLO (秒)
                 scale_out_threshold: float = 1.1,  # 扩容阈值 (10%)
                 scale_in_threshold: float = 0.8,   # 缩容阈值 (20%)
                 scale_out_step: int = 2,           # 扩容步长（每次加 2 个）
                 scale_in_step: int = 1,            # 缩容步长（每次减 1 个）
                 min_instances: int = 1,
                 max_instances: int = 100,
                 **kwargs):
        super().__init__(name="PureLatency", **kwargs)
        
        self.tbt_slo = tbt_slo
        self.scale_out_threshold = scale_out_threshold
        self.scale_in_threshold = scale_in_threshold
        self.scale_out_step = scale_out_step
        self.scale_in_step = scale_in_step
        self.min_instances = min_instances
        self.max_instances = max_instances
    
    def decide(self, metrics: Dict) -> ScalingAction:
        """
        Pure Latency 决策算法：只看延迟，步进式扩缩容
        
        逻辑：
        - TBT > SLO × 1.1 → 加 Step 个实例
        - TBT < SLO × 0.8 → 减 Step 个实例
        """
        current_prompt_instances = metrics.get('current_prompt_instances', 0)
        current_token_instances = metrics.get('current_token_instances', 1)
        tbt = metrics.get('tbt', 0.0)  # Time Between Tokens
        
        if self.debug:
            self.logger.debug(
                f"[PureLatency] tbt={tbt:.3f}s, slo={self.tbt_slo:.3f}s, "
                f"current: P={current_prompt_instances}, T={current_token_instances}"
            )
        
        # 扩容条件：TBT > SLO × 1.1
        if tbt > self.tbt_slo * self.scale_out_threshold and self.can_scale_out():
            # 步进式扩容：每次加 Step 个实例
            target_prompt = current_prompt_instances + self.scale_out_step
            target_token = current_token_instances + self.scale_out_step
            
            # 应用上限
            target_prompt = min(target_prompt, self.max_instances)
            target_token = min(target_token, self.max_instances)
            
            self.record_scale_out()
            
            reason = (f"tbt={tbt:.3f}s > {self.tbt_slo * self.scale_out_threshold:.3f}s "
                     f"(+{self.scale_out_step} instances)")
            
            return ScalingAction(
                decision=ScalingDecision.SCALE_OUT,
                target_prompt_instances=target_prompt,
                target_token_instances=target_token,
                reason=reason
            )
        
        # 缩容条件：TBT < SLO × 0.8
        elif tbt < self.tbt_slo * self.scale_in_threshold and tbt > 0 and self.can_scale_in():
            # 步进式缩容：每次减 Step 个实例
            target_prompt = max(0, current_prompt_instances - self.scale_in_step)
            target_token = max(self.min_instances, current_token_instances - self.scale_in_step)
            
            self.record_scale_in()
            
            reason = (f"tbt={tbt:.3f}s < {self.tbt_slo * self.scale_in_threshold:.3f}s "
                     f"(-{self.scale_in_step} instances)")
            
            return ScalingAction(
                decision=ScalingDecision.SCALE_IN,
                target_prompt_instances=target_prompt,
                target_token_instances=target_token,
                reason=reason
            )
        
        # 维持不变
        else:
            return ScalingAction(
                decision=ScalingDecision.NO_CHANGE,
                target_prompt_instances=current_prompt_instances,
                target_token_instances=current_token_instances,
                reason="within threshold"
            )


class LatencyBasedPolicy(AutoscalingPolicy):
    """
    基于延迟的扩缩容策略（SLO-based）- 多级阈值版本
    
    使用 TTFT 或 TBT 作为指标，采用负反馈控制（多级阈值）
    
    注意：
    - 延迟是非线性的（在瓶颈前平稳，突破后垂直飙升）
    - 不能使用比例公式（会导致资源爆炸）
    - 使用阶梯式扩缩容
    """
    
    def __init__(self,
                 target_latency: float = 1.0,       # 目标延迟（秒）
                 latency_type: str = "ttft",        # "ttft" 或 "tbt"
                 panic_threshold: float = 1.2,      # 严重超标阈值 (20%)
                 scale_out_threshold: float = 1.1,  # 扩容阈值 (10%)
                 scale_in_threshold: float = 0.9,   # 缩容阈值 (10%)
                 panic_scale_factor: float = 1.2,   # 严重超标扩容倍数 (20%)
                 scale_out_factor: float = 1.1,     # 扩容倍数 (10%)
                 scale_in_factor: float = 0.95,     # 缩容倍数 (5%)
                 min_instances: int = 1,
                 max_instances: int = 100,
                 **kwargs):
        name = f"Latency-{latency_type}"
        super().__init__(name=name, **kwargs)
        
        self.target_latency = target_latency
        self.latency_type = latency_type
        self.panic_threshold = panic_threshold
        self.scale_out_threshold = scale_out_threshold
        self.scale_in_threshold = scale_in_threshold
        self.panic_scale_factor = panic_scale_factor
        self.scale_out_factor = scale_out_factor
        self.scale_in_factor = scale_in_factor
        self.min_instances = min_instances
        self.max_instances = max_instances
    
    def decide(self, metrics: Dict) -> ScalingAction:
        """
        基于延迟的负反馈控制算法（多级阈值）
        
        严重超标: L_curr >= L_target × 1.2 → I_new = I_curr × 1.2
        轻微超标: L_curr >= L_target × 1.1 → I_new = I_curr × 1.1
        低负载:   L_curr <= L_target × 0.9 → I_new = I_curr × 0.95
        """
        current_prompt_instances = metrics.get('current_prompt_instances', 0)
        current_token_instances = metrics.get('current_token_instances', 1)
        
        # 获取延迟指标
        if self.latency_type == "ttft":
            current_latency = metrics.get('ttft', 0.0)
            metric_key = 'ttft'
            # TTFT 主要受 Prefill 影响
            target_instances_type = "prompt"
        else:  # tbt
            current_latency = metrics.get('tbt', 0.0)
            metric_key = 'tbt'
            # TBT 主要受 Decode 影响
            target_instances_type = "token"
        
        if self.debug:
            self.logger.debug(
                f"[{self.name}] {metric_key}={current_latency:.3f}s, "
                f"target={self.target_latency:.3f}s"
            )
        
        # 多级阈值决策
        
        # 严重超标 (Panic Scale Up)
        if current_latency >= self.target_latency * self.panic_threshold and self.can_scale_out():
            if target_instances_type == "prompt":
                target_prompt = int(round(current_prompt_instances * self.panic_scale_factor))
                target_prompt = max(self.min_instances, min(target_prompt, self.max_instances))
                target_token = current_token_instances
            else:
                target_prompt = current_prompt_instances
                target_token = int(round(current_token_instances * self.panic_scale_factor))
                target_token = max(self.min_instances, min(target_token, self.max_instances))
            
            self.record_scale_out()
            
            reason = f"PANIC: {metric_key}={current_latency:.3f}s >> target={self.target_latency:.3f}s"
            
            return ScalingAction(
                decision=ScalingDecision.SCALE_OUT,
                target_prompt_instances=target_prompt,
                target_token_instances=target_token,
                reason=reason
            )
        
        # 轻微超标 (Scale Up)
        elif current_latency >= self.target_latency * self.scale_out_threshold and self.can_scale_out():
            if target_instances_type == "prompt":
                target_prompt = int(round(current_prompt_instances * self.scale_out_factor))
                target_prompt = max(self.min_instances, min(target_prompt, self.max_instances))
                target_token = current_token_instances
            else:
                target_prompt = current_prompt_instances
                target_token = int(round(current_token_instances * self.scale_out_factor))
                target_token = max(self.min_instances, min(target_token, self.max_instances))
            
            self.record_scale_out()
            
            reason = f"{metric_key}={current_latency:.3f}s > target={self.target_latency:.3f}s"
            
            return ScalingAction(
                decision=ScalingDecision.SCALE_OUT,
                target_prompt_instances=target_prompt,
                target_token_instances=target_token,
                reason=reason
            )
        
        # 低负载 (Scale Down)
        elif current_latency <= self.target_latency * self.scale_in_threshold and self.can_scale_in():
            if target_instances_type == "prompt":
                target_prompt = int(round(current_prompt_instances * self.scale_in_factor))
                target_prompt = max(self.min_instances, min(target_prompt, self.max_instances))
                target_token = current_token_instances
            else:
                target_prompt = current_prompt_instances
                target_token = int(round(current_token_instances * self.scale_in_factor))
                target_token = max(self.min_instances, min(target_token, self.max_instances))
            
            self.record_scale_in()
            
            reason = f"{metric_key}={current_latency:.3f}s < target={self.target_latency:.3f}s"
            
            return ScalingAction(
                decision=ScalingDecision.SCALE_IN,
                target_prompt_instances=target_prompt,
                target_token_instances=target_token,
                reason=reason
            )
        
        # 维持不变
        else:
            return ScalingAction(
                decision=ScalingDecision.NO_CHANGE,
                target_prompt_instances=current_prompt_instances,
                target_token_instances=current_token_instances,
                reason="within threshold"
            )


class PeriodicPolicy(AutoscalingPolicy):
    """
    周期性策略（Cron-based）
    
    根据时间表硬编码实例数
    例如：
    - 09:00 - 23:00 → 维持 100 台机器
    - 23:00 - 09:00 → 维持 20 台机器
    
    用于对比非动态策略的资源浪费情况
    """
    
    def __init__(self,
                 schedule: List[Tuple[Tuple[int, int], Tuple[int, int], int, int]] = None,
                 **kwargs):
        """
        Args:
            schedule: 时间表列表，格式为 [((start_hour, start_min), (end_hour, end_min), prompt_instances, token_instances)]
                     例如：[((9, 0), (23, 0), 30, 100), ((23, 0), (9, 0), 10, 20)]
        """
        super().__init__(name="Periodic", **kwargs)
        
        if schedule is None:
            # 默认时间表
            self.schedule = [
                ((9, 0), (23, 0), 30, 100),   # 白天高峰
                ((23, 0), (9, 0), 10, 20),    # 夜间低谷
            ]
        else:
            self.schedule = schedule
    
    def decide(self, metrics: Dict) -> ScalingAction:
        """
        根据当前时间查找时间表，返回对应的实例数
        """
        current_time = metrics.get('timestamp', clock())
        
        # 计算当前时间对应的小时和分钟（假设一天 86400 秒）
        day_seconds = current_time % 86400
        hour = int(day_seconds // 3600)
        minute = int((day_seconds % 3600) // 60)
        
        # 查找匹配的时间段
        for (start_hour, start_min), (end_hour, end_min), prompt_inst, token_inst in self.schedule:
            start_minutes = start_hour * 60 + start_min
            end_minutes = end_hour * 60 + end_min
            current_minutes = hour * 60 + minute
            
            # 处理跨天的情况（例如 23:00 - 09:00）
            if start_minutes > end_minutes:
                if current_minutes >= start_minutes or current_minutes < end_minutes:
                    return ScalingAction(
                        decision=ScalingDecision.NO_CHANGE,
                        target_prompt_instances=prompt_inst,
                        target_token_instances=token_inst,
                        reason=f"scheduled: {start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}"
                    )
            else:
                if start_minutes <= current_minutes < end_minutes:
                    return ScalingAction(
                        decision=ScalingDecision.NO_CHANGE,
                        target_prompt_instances=prompt_inst,
                        target_token_instances=token_inst,
                        reason=f"scheduled: {start_hour:02d}:{start_min:02d}-{end_hour:02d}:{end_min:02d}"
                    )
        
        # 如果没有匹配，返回默认值
        current_prompt = metrics.get('current_prompt_instances', 0)
        current_token = metrics.get('current_token_instances', 1)
        return ScalingAction(
            decision=ScalingDecision.NO_CHANGE,
            target_prompt_instances=current_prompt,
            target_token_instances=current_token,
            reason="no schedule match"
        )


class RLPolicy(AutoscalingPolicy):
    """
    强化学习策略（预留接口）
    
    用于实现基于强化学习的自动扩缩容
    继承此类并实现具体的 RL 算法（如 DQN, PPO, A3C 等）
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 state_dim: int = 10,
                 action_dim: int = 5,
                 **kwargs):
        super().__init__(name="RL", **kwargs)
        
        self.model_path = model_path
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 用于存储模型（由子类实现）
        self.model = None
    
    def decide(self, metrics: Dict) -> ScalingAction:
        """
        使用强化学习模型做决策
        
        子类应该实现：
        1. 将 metrics 转换为 state
        2. 调用 model 推理得到 action
        3. 将 action 转换为 ScalingAction
        """
        # 默认实现：维持不变
        return ScalingAction(
            decision=ScalingDecision.NO_CHANGE,
            target_prompt_instances=metrics.get('current_prompt_instances', 0),
            target_token_instances=metrics.get('current_token_instances', 1),
            reason="RL model not implemented"
        )
    
    def load_model(self, model_path: str):
        """加载 RL 模型"""
        raise NotImplementedError("Subclass should implement load_model()")
    
    def save_model(self, model_path: str):
        """保存 RL 模型"""
        raise NotImplementedError("Subclass should implement save_model()")


class NoAutoscalingPolicy(AutoscalingPolicy):
    """
    不进行自动扩缩容的策略（固定实例数）
    
    用于对比实验
    """
    
    def __init__(self,
                 fixed_prompt_instances: int = 1,
                 fixed_token_instances: int = 3,
                 **kwargs):
        super().__init__(name="NoAutoscaling", **kwargs)
        
        self.fixed_prompt_instances = fixed_prompt_instances
        self.fixed_token_instances = fixed_token_instances
    
    def decide(self, metrics: Dict) -> ScalingAction:
        """始终返回固定的实例数"""
        return ScalingAction(
            decision=ScalingDecision.NO_CHANGE,
            target_prompt_instances=self.fixed_prompt_instances,
            target_token_instances=self.fixed_token_instances,
            reason="fixed instances"
        )

