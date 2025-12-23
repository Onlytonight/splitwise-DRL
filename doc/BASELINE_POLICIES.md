# 自动扩缩容基线策略文档

本文档详细说明了实现的各种基线（Baseline）扩缩容策略，用于与 HeteroScale 进行对比实验。

---

## 概述

为了评估 HeteroScale 的有效性，我们实现了4种常见的自动扩缩容基线策略：

1. **HPA-GPU** (Baseline 1): 基于硬件利用率的传统方式
2. **Independent TPS** (Baseline 2): 基于 TPS 但不协同
3. **Pure Latency** (Baseline 3): 纯延迟驱动/负反馈
4. **Periodic** (Baseline 4): 周期性/定时任务

每种策略都有其典型的失败模式，通过对比可以突出 HeteroScale 的优势。

---

## Baseline 1: HPA-GPU（基于硬件利用率）

### 简介

这是最常见的 Kubernetes HPA（Horizontal Pod Autoscaler）默认方式。

### 核心逻辑

- **监控信号**: Prefill 池和 Decode 池分别监控自己的 GPU 利用率（显存使用率）
- **扩缩公式**: 
  $$I_{target} = \lceil I_{curr} \times \frac{Util_{observed}}{Util_{target}} \rceil$$
- **独立性**: Prefill 池和 Decode 池完全独立，互不通信

### 实现细节

```python
class HPAGPUPolicy(AutoscalingPolicy):
    def decide(self, metrics: Dict) -> ScalingAction:
        # Prefill 池独立决策
        expected_prompt = current_prompt_instances * (prefill_util / target_prefill_util)
        
        # Decode 池独立决策
        expected_token = current_token_instances * (decode_util / target_decode_util)
        
        # 判断扩缩容
        if ratio > threshold:
            scale_out()
        elif ratio < threshold:
            scale_in()
```

### 预期失败点

**Decode 池即使在没流量时也不缩容**

- **原因**: Decode 节点的 GPU 利用率受 KV Cache 显存管理影响
- **表现**: 即使负载低，利用率也常年维持在 80%-90%（Memory-bound）
- **后果**: 资源浪费严重，无法有效缩容

### 配置文件

```yaml
# configs/autoscaling_policy/hpa_gpu.yaml
_target_: autoscaling_policies.HPAGPUPolicy

target_prefill_util: 0.7   # Prefill 目标利用率 (70%)
target_decode_util: 0.7    # Decode 目标利用率 (70%)

scale_out_threshold: 0.1   # 扩容阈值 (10%)
scale_in_threshold: 0.1    # 缩容阈值 (10%)

min_instances: 1
max_instances: 100

scale_out_cooldown: 180.0  # 3分钟
scale_in_cooldown: 600.0   # 10分钟
```

### 使用方法

```bash
# 修改 config.yaml 中的 autoscaling_policy
autoscaling_policy: hpa_gpu

# 或在命令行指定
python run.py autoscaling_policy=hpa_gpu
```

---

## Baseline 2: Independent TPS（独立 TPS 扩缩容）

### 简介

这是"看起来科学"但实际效果差的方法。两个池子各自监控自己的 TPS，互不协调。

### 核心逻辑

- **监控信号**:
  - Prefill 池监控自己的 `Prefill_TPS`
  - Decode 池监控自己的 `Decode_TPS`
- **扩缩公式**:
  - $I_{p\_target} = I_{p\_curr} \times (Prefill\_TPS / P\_TPS\_Target)$
  - $I_{d\_target} = I_{d\_curr} \times (Decode\_TPS / D\_TPS\_Target)$
- **独立性**: P 池只管自己的 TPS，D 池只管自己的 TPS

### 实现细节

```python
class IndependentTPSPolicy(AutoscalingPolicy):
    def decide(self, metrics: Dict) -> ScalingAction:
        # Prefill 池独立决策
        prefill_tps_per_instance = prefill_tps / current_prompt_instances
        expected_prompt = current_prompt_instances * (prefill_tps_per_instance / target_prefill_tps)
        
        # Decode 池独立决策
        decode_tps_per_instance = decode_tps / current_token_instances
        expected_token = current_token_instances * (decode_tps_per_instance / target_decode_tps)
        
        # 独立判断扩缩容
        if prompt_ratio > threshold or token_ratio > threshold:
            scale_out()
        elif prompt_ratio < threshold and token_ratio < threshold:
            scale_in()
```

### 预期失败点

**P/D 比例剧烈抖动**

- **原因**: 两个池子扩缩容速度不一致
- **场景示例**:
  1. 流量增加时，P 池扩容快而 D 池扩容慢
  2. P 池处理了大量 Prompt 塞给 D 池
  3. D 池接不住，导致 TBT 飙升
  4. D 池开始扩容，但此时 P 池可能已经开始缩容
  5. 循环往复，比例不断波动
- **后果**: 资源利用率低，延迟不稳定

### 配置文件

```yaml
# configs/autoscaling_policy/independent_tps.yaml
_target_: autoscaling_policies.IndependentTPSPolicy

target_prefill_tps: 10.0   # 单 Prefill 实例目标 TPS
target_decode_tps: 100.0   # 单 Decode 实例目标 TPS

scale_out_threshold: 0.1
scale_in_threshold: 0.1

min_instances: 1
max_instances: 100

scale_out_cooldown: 180.0
scale_in_cooldown: 600.0
```

### 使用方法

```bash
python run.py autoscaling_policy=independent_tps
```

---

## Baseline 3: Pure Latency（纯延迟驱动）

### 简介

不参考吞吐量，只看用户卡不卡的反应式方法。使用步进式扩缩容。

### 核心逻辑

- **监控信号**: P90 TBT (Time Between Tokens)
- **扩缩逻辑**:
  - If `TBT > SLO × 1.1`: `Count = Count + Step` (例如每次加 2 个 Pod)
  - If `TBT < SLO × 0.8`: `Count = Count - Step` (例如每次减 1 个 Pod)
- **特点**: 步进式（Step-based），不使用比例公式

### 实现细节

```python
class PureLatencyPolicy(AutoscalingPolicy):
    def decide(self, metrics: Dict) -> ScalingAction:
        tbt = metrics.get('tbt', 0.0)
        
        # 扩容条件：TBT > SLO × 1.1
        if tbt > self.tbt_slo * self.scale_out_threshold:
            # 步进式扩容：每次加 Step 个实例
            target_prompt = current_prompt_instances + self.scale_out_step
            target_token = current_token_instances + self.scale_out_step
            return scale_out()
        
        # 缩容条件：TBT < SLO × 0.8
        elif tbt < self.tbt_slo * self.scale_in_threshold:
            # 步进式缩容：每次减 Step 个实例
            target_prompt = current_prompt_instances - self.scale_in_step
            target_token = current_token_instances - self.scale_in_step
            return scale_in()
```

### 预期失败点

**严重的扩缩容振荡（Flapping）**

- **原因**: 延迟和 Pod 数量是非线性关系
- **场景示例**:
  1. TBT > SLO × 1.1，触发扩容，加 2 个 Pod
  2. 加了 2 个 Pod 后，延迟可能瞬间降到极低（例如 TBT = SLO × 0.5）
  3. TBT < SLO × 0.8，触发缩容，减 1 个 Pod
  4. 缩容后延迟又瞬间飙升
  5. 系统不停地在"增-删-增"之间跳变
- **后果**: 资源利用率极低，用户体验差

### 配置文件

```yaml
# configs/autoscaling_policy/pure_latency.yaml
_target_: autoscaling_policies.PureLatencyPolicy

tbt_slo: 0.04  # TBT SLO (秒)

scale_out_threshold: 1.1   # 扩容阈值 (TBT > SLO × 1.1)
scale_in_threshold: 0.8    # 缩容阈值 (TBT < SLO × 0.8)

scale_out_step: 2          # 扩容步长（每次加 2 个实例）
scale_in_step: 1           # 缩容步长（每次减 1 个实例）

min_instances: 1
max_instances: 100

scale_out_cooldown: 180.0
scale_in_cooldown: 600.0
```

### 使用方法

```bash
python run.py autoscaling_policy=pure_latency
```

---

## Baseline 4: Periodic（周期性/定时任务）

### 简介

这是字节跳动在没上 HeteroScale 之前的备选方案。根据时间表硬编码实例数。

### 核心逻辑

- **实现方式**: Cron Job
- **逻辑**:
  - 09:00 - 23:00（高峰期）：设置 $I_p = 30, I_d = 100$
  - 23:00 - 09:00（低峰期）：设置 $I_p = 10, I_d = 20$

### 实现细节

```python
class PeriodicPolicy(AutoscalingPolicy):
    def decide(self, metrics: Dict) -> ScalingAction:
        current_time = metrics.get('timestamp', clock())
        
        # 计算当前时间对应的小时和分钟
        hour = int((current_time % 86400) // 3600)
        
        # 查找匹配的时间段
        for (start_hour, start_min), (end_hour, end_min), prompt_inst, token_inst in self.schedule:
            if is_in_time_range(hour, start_hour, end_hour):
                return ScalingAction(
                    target_prompt_instances=prompt_inst,
                    target_token_instances=token_inst
                )
```

### 预期失败点

**无法应对突发流量（Flash Crowd）**

- **原因**: 固定时间表，不感知实时负载
- **场景示例**:
  - 如果下午 2 点突然有一个热点事件导致流量暴增
  - 周期性缩放无法感知，系统继续维持 100 台机器
  - 如果流量远超预期，系统会直接崩溃
  - 如果流量低于预期，资源严重浪费
- **后果**: 无弹性，资源利用率低

### 配置文件

```yaml
# configs/autoscaling_policy/periodic.yaml
_target_: autoscaling_policies.PeriodicPolicy

# 时间表：[(start_time, end_time, prompt_instances, token_instances), ...]
schedule:
  - [[9, 0], [23, 0], 30, 100]    # 白天高峰 09:00-23:00
  - [[23, 0], [9, 0], 10, 20]     # 夜间低谷 23:00-09:00

scale_out_cooldown: 0.0  # 周期性策略不需要冷却时间
scale_in_cooldown: 0.0
```

### 使用方法

```bash
python run.py autoscaling_policy=periodic
```

---

## 对比总结

| 策略 | 监控指标 | 扩缩容算法 | 主要问题 | 适用场景 |
|-----|---------|-----------|---------|---------|
| **HPA-GPU** | GPU 利用率 | 比例控制 | Decode 池无法缩容 | Prefill 为主的场景 |
| **Independent TPS** | Prefill TPS + Decode TPS | 比例控制（独立） | P/D 比例抖动 | 流量平稳的场景 |
| **Pure Latency** | TBT | 步进式 | 严重振荡 | 对延迟非常敏感的场景 |
| **Periodic** | 时间表 | 固定调度 | 无法应对突发流量 | 流量规律性极强的场景 |
| **HeteroScale** | Decode TPS + TBT | 比例控制 + 负反馈 | 无 | 所有场景 |

---

## 实验对比建议

### 实验设置

1. **流量模式**:
   - 平稳流量：测试基本扩缩容能力
   - 波动流量：测试反应速度
   - 突发流量：测试应对能力

2. **评估指标**:
   - **资源利用率**: 平均 GPU 利用率
   - **延迟 SLO 违规率**: TTFT/TBT 超过 SLO 的比例
   - **扩缩容次数**: 越少越好（稳定性）
   - **P/D 比例方差**: 越小越好（协同性）

3. **实验步骤**:
   ```bash
   # 运行各个基线
   python run.py autoscaling_policy=hpa_gpu trace=azure_trace
   python run.py autoscaling_policy=independent_tps trace=azure_trace
   python run.py autoscaling_policy=pure_latency trace=azure_trace
   python run.py autoscaling_policy=periodic trace=azure_trace
   
   # 运行 HeteroScale
   python run.py autoscaling_policy=heteroscale trace=azure_trace
   
   # 分析结果
   python analyze_results.py --compare-policies
   ```

### 预期结果

- **HPA-GPU**: 资源浪费严重（Decode 池不缩容）
- **Independent TPS**: P/D 比例不稳定，延迟波动大
- **Pure Latency**: 扩缩容次数极多，振荡严重
- **Periodic**: 无法适应流量变化，SLO 违规率高或资源浪费
- **HeteroScale**: 各指标最优，P/D 比例稳定

---

## 扩展：自定义基线策略

如果需要实现自己的基线策略，可以继承 `AutoscalingPolicy` 基类：

```python
from autoscaling_policies import AutoscalingPolicy, ScalingAction, ScalingDecision

class MyCustomPolicy(AutoscalingPolicy):
    def __init__(self, custom_param: float = 1.0, **kwargs):
        super().__init__(name="MyCustom", **kwargs)
        self.custom_param = custom_param
    
    def decide(self, metrics: Dict) -> ScalingAction:
        # 实现你的决策逻辑
        current_prompt = metrics.get('current_prompt_instances', 0)
        current_token = metrics.get('current_token_instances', 1)
        
        # ... 你的算法 ...
        
        return ScalingAction(
            decision=ScalingDecision.SCALE_OUT,  # 或 SCALE_IN, NO_CHANGE
            target_prompt_instances=target_prompt,
            target_token_instances=target_token,
            reason="your reason here"
        )
```

然后创建配置文件 `configs/autoscaling_policy/my_custom.yaml`：

```yaml
_target_: autoscaling_policies.MyCustomPolicy
custom_param: 1.0
scale_out_cooldown: 180.0
scale_in_cooldown: 600.0
```

---

## 参考文献

- HeteroScale 论文: [链接]
- Kubernetes HPA 文档: https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/

