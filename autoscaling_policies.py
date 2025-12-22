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
    HeteroScale 策略 (TPS-based)
    
    基于 Decode TPS 的比例控制，维持固定的 Prefill/Decode 比例
    
    核心思想：
    1. 使用 Decode TPS 作为负载指标（信噪比最高、线性相关）
    2. 比例控制算法计算期望实例数
    3. 强制维持固定的 P/D 比例
    
    参考：HeteroScale 论文 Algorithm 2
    """
    
    def __init__(self,
                 target_decode_tps_per_instance: float = 100.0,  # 单实例目标 Decode TPS
                 pd_ratio: float = 0.33,  # P/D 比例 (Prefill / Decode)
                 scale_out_threshold: float = 0.1,  # 扩容阈值 (10%)
                 scale_in_threshold: float = 0.1,   # 缩容阈值 (10%)
                 min_instances: int = 1,            # 最小实例数
                 max_instances: int = 100,          # 最大实例数
                 **kwargs):
        super().__init__(name="HeteroScale", **kwargs)
        
        self.target_decode_tps_per_instance = target_decode_tps_per_instance
        self.pd_ratio = pd_ratio
        self.scale_out_threshold = scale_out_threshold
        self.scale_in_threshold = scale_in_threshold
        self.min_instances = min_instances
        self.max_instances = max_instances
    
    def decide(self, metrics: Dict) -> ScalingAction:
        """
        基于 Decode TPS 的比例控制算法
        
        公式：
        I_expected = M_current_total / M_target
        R = I_expected / I_current
        
        扩容条件: R > 1 + θ_out
        缩容条件: R < 1 - θ_in
        """
        current_token_instances = metrics.get('current_token_instances', 1)
        current_prompt_instances = metrics.get('current_prompt_instances', 0)
        decode_tps_total = metrics.get('decode_tps', 0.0)
        
        # 防止除零
        if current_token_instances == 0:
            current_token_instances = 1
        
        # 计算期望的 Decode 实例数
        # I_expected = M_total / M_target
        expected_token_instances = decode_tps_total / self.target_decode_tps_per_instance
        
        # 计算变化率
        # R = I_expected / I_current
        ratio = expected_token_instances / current_token_instances
        
        if self.debug:
            self.logger.debug(
                f"[HeteroScale] decode_tps={decode_tps_total:.2f}, "
                f"current_token={current_token_instances}, "
                f"expected_token={expected_token_instances:.2f}, "
                f"ratio={ratio:.3f}"
            )
        
        # 决策逻辑
        # 扩容条件: R > 1 + θ_out
        if ratio > (1 + self.scale_out_threshold) and self.can_scale_out():
            target_token = int(round(expected_token_instances))
            target_token = max(self.min_instances, min(target_token, self.max_instances))
            
            # 根据 P/D 比例计算 Prefill 实例数
            target_prompt = int(round(target_token * self.pd_ratio))
            target_prompt = max(0, min(target_prompt, self.max_instances))
            
            self.record_scale_out()
            
            reason = (f"decode_tps={decode_tps_total:.1f} > "
                     f"target={current_token_instances * self.target_decode_tps_per_instance:.1f}")
            
            return ScalingAction(
                decision=ScalingDecision.SCALE_OUT,
                target_prompt_instances=target_prompt,
                target_token_instances=target_token,
                reason=reason
            )
        
        # 缩容条件: R < 1 - θ_in
        elif ratio < (1 - self.scale_in_threshold) and self.can_scale_in():
            target_token = int(round(expected_token_instances))
            target_token = max(self.min_instances, min(target_token, self.max_instances))
            
            # 根据 P/D 比例计算 Prefill 实例数
            target_prompt = int(round(target_token * self.pd_ratio))
            target_prompt = max(0, min(target_prompt, self.max_instances))
            
            self.record_scale_in()
            
            reason = (f"decode_tps={decode_tps_total:.1f} < "
                     f"target={current_token_instances * self.target_decode_tps_per_instance:.1f}")
            
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


class UtilizationBasedPolicy(AutoscalingPolicy):
    """
    基于资源利用率的扩缩容策略（类似 Kubernetes HPA）
    
    使用 GPU 利用率作为指标，采用比例控制算法
    
    注意：
    - Prefill Util 与负载相关性较好
    - Decode Util 通常失效（Memory-bound，即使低负载也可能显示高利用率）
    """
    
    def __init__(self,
                 target_utilization: float = 0.7,   # 目标利用率 (70%)
                 utilization_type: str = "prefill", # "prefill" 或 "decode"
                 scale_out_threshold: float = 0.1,  # 扩容阈值
                 scale_in_threshold: float = 0.1,   # 缩容阈值
                 min_instances: int = 1,
                 max_instances: int = 100,
                 **kwargs):
        name = f"Utilization-{utilization_type}"
        super().__init__(name=name, **kwargs)
        
        self.target_utilization = target_utilization
        self.utilization_type = utilization_type
        self.scale_out_threshold = scale_out_threshold
        self.scale_in_threshold = scale_in_threshold
        self.min_instances = min_instances
        self.max_instances = max_instances
    
    def decide(self, metrics: Dict) -> ScalingAction:
        """
        基于利用率的比例控制算法
        
        公式：
        I_expected = I_current × (Util_current / Util_target)
        """
        if self.utilization_type == "prefill":
            current_instances = metrics.get('current_prompt_instances', 1)
            current_util = metrics.get('prefill_gpu_util', 0.0)
            metric_key = 'prefill_gpu_util'
        else:  # decode
            current_instances = metrics.get('current_token_instances', 1)
            current_util = metrics.get('decode_gpu_util', 0.0)
            metric_key = 'decode_gpu_util'
        
        if current_instances == 0:
            current_instances = 1
        
        # 计算期望实例数
        # I_expected = I_current × (Util_current / Util_target)
        expected_instances = current_instances * (current_util / self.target_utilization)
        
        # 计算变化率
        ratio = expected_instances / current_instances
        
        if self.debug:
            self.logger.debug(
                f"[{self.name}] util={current_util:.2f}, "
                f"current={current_instances}, "
                f"expected={expected_instances:.2f}, "
                f"ratio={ratio:.3f}"
            )
        
        # 扩容
        if ratio > (1 + self.scale_out_threshold) and self.can_scale_out():
            target = int(round(expected_instances))
            target = max(self.min_instances, min(target, self.max_instances))
            
            self.record_scale_out()
            
            reason = f"{metric_key}={current_util:.2f} > target={self.target_utilization:.2f}"
            
            if self.utilization_type == "prefill":
                return ScalingAction(
                    decision=ScalingDecision.SCALE_OUT,
                    target_prompt_instances=target,
                    target_token_instances=metrics.get('current_token_instances', 1),
                    reason=reason
                )
            else:
                return ScalingAction(
                    decision=ScalingDecision.SCALE_OUT,
                    target_prompt_instances=metrics.get('current_prompt_instances', 0),
                    target_token_instances=target,
                    reason=reason
                )
        
        # 缩容
        elif ratio < (1 - self.scale_in_threshold) and self.can_scale_in():
            target = int(round(expected_instances))
            target = max(self.min_instances, min(target, self.max_instances))
            
            self.record_scale_in()
            
            reason = f"{metric_key}={current_util:.2f} < target={self.target_utilization:.2f}"
            
            if self.utilization_type == "prefill":
                return ScalingAction(
                    decision=ScalingDecision.SCALE_IN,
                    target_prompt_instances=target,
                    target_token_instances=metrics.get('current_token_instances', 1),
                    reason=reason
                )
            else:
                return ScalingAction(
                    decision=ScalingDecision.SCALE_IN,
                    target_prompt_instances=metrics.get('current_prompt_instances', 0),
                    target_token_instances=target,
                    reason=reason
                )
        
        # 维持不变
        else:
            return ScalingAction(
                decision=ScalingDecision.NO_CHANGE,
                target_prompt_instances=metrics.get('current_prompt_instances', 0),
                target_token_instances=metrics.get('current_token_instances', 1),
                reason="within threshold"
            )


class LatencyBasedPolicy(AutoscalingPolicy):
    """
    基于延迟的扩缩容策略（SLO-based）
    
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

