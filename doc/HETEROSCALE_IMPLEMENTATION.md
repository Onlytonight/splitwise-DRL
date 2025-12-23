# HeteroScale 策略实现说明

## 概述

HeteroScale 是生产环境推荐的自动扩缩容策略，结合了**比例控制**和**延迟触发**两种机制，既能处理日常流量波动，又能快速响应突发流量。

## 核心机制

### 1. 比例控制（主推力）

用于处理日常流量波动，平稳扩缩容。

#### 输入参数
- `TPS_curr`：当前总解码 TPS（所有 decode 实例的总和）
- `TPS_target`：单实例目标 TPS
- `I_curr`：当前总实例数

#### 计算公式

**步骤 1：计算目标总实例数**

```python
I_new_total = TPS_curr / TPS_target
```

**步骤 2：维持 P/D 比例分配**

假设 P/D 比例为 1:3（`pd_ratio = 0.33`）：

```python
# 总份数 = 1 + 3 = 4
total_parts = 1 + (1 / pd_ratio) = 1 + 3 = 4

# Prefill 实例数 = 总数 / 4
I_prefill = ⌈I_new_total / 4⌉ = ⌈I_new_total × 0.25⌉

# Decode 实例数 = Prefill × 3
I_decode = I_prefill × 3
```

**步骤 3：判断是否扩缩容**

```python
ratio = I_new_total / I_curr

if ratio > 1 + θ_out:     # 例如 > 1.1
    执行扩容
elif ratio < 1 - θ_in:    # 例如 < 0.9
    执行缩容
else:
    维持不变
```

### 2. 延迟触发（安全网）

用于应对突发流量，快速扩容。

#### 监控信号
- `TBT`（Time Between Tokens）：字间延迟

#### 触发逻辑

```python
if TBT > 1.2 × SLO:
    # 紧急扩容：跳跃性增加，不经过比例计算
    I_prefill_new = I_prefill_curr × 1.2  # 扩容 20%
    I_decode_new = I_decode_curr × 1.2    # 扩容 20%
```

**特点**：
- ✅ **优先级最高**：在比例控制之前检查
- ✅ **快速响应**：直接按百分比扩容
- ✅ **跳跃性**：不经过比例计算，避免延迟

## 实现细节

### 完整决策流程

```
┌─────────────────────────────────────────────────────────┐
│ HeteroScalePolicy.decide()                              │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. 延迟触发检查（安全网）                              │
│     ├─> TBT > 1.2 × SLO？                               │
│     │    ├─ 是 → 紧急扩容 20% ⚡                        │
│     │    └─ 否 → 继续下一步                            │
│     │                                                    │
│  2. 比例控制（主推力）                                  │
│     ├─> 计算目标总实例数                               │
│     │    I_new_total = TPS_curr / TPS_target           │
│     │                                                    │
│     ├─> 根据 P/D 比例分配                              │
│     │    total_parts = 1 + (1 / pd_ratio)              │
│     │    I_prefill = I_new_total / total_parts         │
│     │    I_decode = I_prefill × (1 / pd_ratio)         │
│     │                                                    │
│     ├─> 计算变化率                                      │
│     │    ratio = I_new_total / I_curr                   │
│     │                                                    │
│     └─> 决策                                            │
│          ├─ ratio > 1.1 → 扩容 📈                       │
│          ├─ ratio < 0.9 → 缩容 📉                       │
│          └─ 否则 → 维持不变 ➡️                         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 代码示例

```python
# 1. 延迟触发检查
if enable_latency_trigger and tbt > tbt_slo * 1.2:
    # 紧急扩容
    target_prompt = int(current_prompt * 1.2)
    target_token = int(current_token * 1.2)
    return SCALE_OUT

# 2. 比例控制
total_needed = decode_tps / target_tps_per_instance
total_parts = 1 + (1 / pd_ratio)  # 1 + 3 = 4
expected_prompt = total_needed / total_parts
expected_token = expected_prompt * (1 / pd_ratio)

ratio = (expected_prompt + expected_token) / (current_prompt + current_token)

if ratio > 1.1:
    return SCALE_OUT
elif ratio < 0.9:
    return SCALE_IN
else:
    return NO_CHANGE
```

## 配置参数

### 比例控制参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `target_decode_tps_per_instance` | 100.0 | 单实例目标 TPS |
| `pd_ratio` | 0.33 | P/D 比例（1:3） |
| `scale_out_threshold` | 0.1 | 扩容阈值（10%） |
| `scale_in_threshold` | 0.1 | 缩容阈值（10%） |
| `min_instances` | 1 | 最小实例数 |
| `max_instances` | 100 | 最大实例数 |

### 延迟触发参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `enable_latency_trigger` | true | 是否启用延迟触发 |
| `tbt_slo` | 0.1 | TBT SLO（秒） |
| `latency_panic_threshold` | 1.2 | 延迟恐慌阈值（1.2x） |
| `latency_panic_scale_factor` | 1.2 | 恐慌时扩容倍数（20%） |

### 冷却时间

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `scale_out_cooldown` | 180.0 | 扩容冷却 3 分钟 |
| `scale_in_cooldown` | 600.0 | 缩容冷却 10 分钟 |

## 使用示例

### 基本使用

```bash
# 使用默认参数
python run.py autoscaling_policy=heteroscale autoscaling.enable=true
```

### 自定义参数

```bash
# 调整 TPS 目标和 P/D 比例
python run.py \
  autoscaling_policy=heteroscale \
  autoscaling.enable=true \
  autoscaling_policy.target_decode_tps_per_instance=150.0 \
  autoscaling_policy.pd_ratio=0.5
```

### 禁用延迟触发

```bash
# 只使用比例控制
python run.py \
  autoscaling_policy=heteroscale \
  autoscaling.enable=true \
  autoscaling_policy.enable_latency_trigger=false
```

### 调整延迟触发灵敏度

```bash
# 更激进的延迟触发
python run.py \
  autoscaling_policy=heteroscale \
  autoscaling.enable=true \
  autoscaling_policy.latency_panic_threshold=1.1 \
  autoscaling_policy.latency_panic_scale_factor=1.5
```

## 工作示例

### 场景 1：日常流量波动

**初始状态**：
- Prefill 实例：10 个
- Decode 实例：30 个
- 当前 TPS：2500
- 目标单实例 TPS：100

**计算过程**：
```python
# 目标总实例数
total_needed = 2500 / 100 = 25

# P/D 分配（1:3）
total_parts = 1 + 3 = 4
expected_prefill = 25 / 4 = 6.25 ≈ 6
expected_decode = 6 × 3 = 18

# 变化率
current_total = 10 + 30 = 40
expected_total = 6 + 18 = 24
ratio = 24 / 40 = 0.6

# 决策
ratio < 0.9 → 缩容
```

**结果**：缩容到 6 个 Prefill + 18 个 Decode

### 场景 2：突发流量

**初始状态**：
- Prefill 实例：10 个
- Decode 实例：30 个
- TBT：0.15 秒
- TBT SLO：0.1 秒

**触发延迟安全网**：
```python
# 检查延迟
TBT = 0.15 > 0.1 × 1.2 = 0.12 → 触发紧急扩容

# 直接扩容 20%
new_prefill = 10 × 1.2 = 12
new_decode = 30 × 1.2 = 36
```

**结果**：快速扩容到 12 个 Prefill + 36 个 Decode

## 优势

### 与其他策略对比

| 特性 | HeteroScale | Utilization | Latency | Periodic |
|-----|-------------|-------------|---------|----------|
| 反应速度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| 准确性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| 稳定性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 资源效率 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 突发应对 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ |

### 核心优势

1. **双重保护** 🛡️
   - 比例控制：平稳处理日常波动
   - 延迟触发：快速响应突发流量

2. **线性相关** 📈
   - TPS 与负载线性相关
   - 避免 GPU 利用率的 Memory-bound 问题
   - 避免延迟的非线性特性

3. **P/D 协同** 🤝
   - 强制维持最佳比例
   - 避免某一类实例成为瓶颈
   - 最大化资源利用率

4. **快速响应** ⚡
   - 延迟触发机制确保 SLO 达标
   - 跳跃性扩容，无需等待比例计算

## 调试和监控

### 启用调试模式

```bash
python run.py \
  autoscaling_policy=heteroscale \
  autoscaling.enable=true \
  autoscaling_policy.debug=true
```

### 查看日志

扩缩容日志：`results/<seed>/<config>/scaling/0.log`

```
time,action,target,status
30.0,autoscaling_decision,scale_out,prompt:10->12_token:30->36,LATENCY_PANIC: tbt=0.15s > 0.12s
60.0,autoscaling_decision,scale_in,prompt:12->6_token:36->18,decode_tps=1500.0 < 3000.0
```

### 关键指标

监控以下指标评估策略效果：

1. **扩缩容频率**：过高说明阈值太敏感
2. **资源利用率**：过低说明过度配置
3. **延迟分位数**：p99 < SLO 说明策略有效
4. **TPS 趋势**：应该跟随流量曲线

## 参数调优指南

### 1. 确定目标 TPS

运行压测，观察单实例饱和 TPS：

```bash
# 固定 1 个 decode 实例，逐步增加流量
# 记录 TPS 饱和点
target_decode_tps_per_instance = 饱和点 TPS × 0.7  # 留 30% 余量
```

### 2. 确定 P/D 比例

运行不同比例的实验：

```bash
# 测试 1:2, 1:3, 1:4 等比例
for ratio in 0.5 0.33 0.25; do
    python run.py autoscaling_policy.pd_ratio=$ratio
done

# 选择吞吐量最高的比例
```

### 3. 调整阈值

根据流量波动幅度调整：

- **流量平稳**：增大阈值（0.15-0.2），减少扩缩容
- **流量波动大**：减小阈值（0.05-0.1），快速响应

### 4. 调整冷却时间

根据实例启动时间调整：

- **启动快**（<30s）：缩短冷却时间
- **启动慢**（>60s）：延长冷却时间

## 总结

HeteroScale 是**生产环境推荐**的扩缩容策略，因为：

✅ **精准**：基于 TPS 的线性关系  
✅ **快速**：延迟触发机制应对突发  
✅ **稳定**：比例控制避免震荡  
✅ **高效**：P/D 协同最大化资源利用  
✅ **可靠**：双重保护机制确保 SLO

适用于所有 LLM 推理服务的自动扩缩容场景！🚀

