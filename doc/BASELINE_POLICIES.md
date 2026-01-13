# 基线扩缩容策略使用指南

本文档介绍如何在现有模拟器架构下运行非学习的基线扩缩容策略。

## 概述

基线策略是非学习的规则策略，用于与强化学习方法进行对比。与 RL 方法不同，基线策略：
- ✅ **不需要训练**：单次运行即可
- ✅ **规则明确**：基于明确的阈值和规则做决策
- ✅ **易于调优**：通过调整超参数即可
- ✅ **可解释性强**：每个决策都有明确的触发条件

本框架支持三种基线策略：
1. **HeteroScale** - 生产级策略，结合比例控制和延迟触发
2. **UtilizationBased** - 基于显存利用率的简单策略
3. **QueueBased** - 基于队列长度的简单策略

## 架构设计

基线策略复用了 RL 框架的核心组件：

```
┌─────────────────────────────────────────────────────────────┐
│ TraceBaselineSimulator                                      │
│ (基线模拟器)                                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │ RLStateCollector │      │   BasePolicy     │            │
│  │  (状态收集器)    │─────▶│  (策略对象)      │            │
│  └──────────────────┘      └──────────────────┘            │
│         │                           │                       │
│         │                           ▼                       │
│         │                  ┌─────────────────┐             │
│         │                  │   决策逻辑       │             │
│         │                  │  (规则策略)      │             │
│         │                  └─────────────────┘             │
│         │                           │                       │
│         ▼                           ▼                       │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │ RLRewardCalculator│      │ RLActionExecutor │            │
│  │  (奖励计算器)     │      │  (动作执行器)    │            │
│  └──────────────────┘      └──────────────────┘            │
│         │                           │                       │
│         └───────────┬───────────────┘                       │
│                     ▼                                       │
│              评估和执行                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**关键特性**：
- 复用 `RLStateCollector` 收集系统状态
- 复用 `RLRewardCalculator` 计算奖励（用于评估）
- 复用 `RLActionExecutor` 执行扩缩容动作
- 使用 `BasePolicy` 及其子类实现具体的决策逻辑

## 1. HeteroScale 策略

### 策略原理

HeteroScale 是生产环境推荐的自动扩缩容策略，结合了两种互补的机制：

#### 1.1 比例控制（主推力）

基于 TPS（每秒生成的 token 数）进行平稳的扩缩容：

**计算流程**：
```python
# 步骤 1：计算目标总实例数
target_total = current_tps / target_tps_per_instance

# 步骤 2：根据 P/D 比例分配
# 假设 pd_ratio = 0.33 (即 1:3)
total_parts = 1 + (1 / pd_ratio)  # = 4
target_prefill = target_total / total_parts
target_decode = target_prefill / pd_ratio

# 步骤 3：判断是否扩缩容
ratio = target_total / current_total
if ratio > 1.1:    # 扩容
    scale_out()
elif ratio < 0.9:  # 缩容
    scale_in()
else:              # 维持不变
    maintain()
```

#### 1.2 延迟触发（安全网）

基于 TBT（Time Between Tokens）进行紧急扩容：

```python
if tbt > 1.2 × tbt_slo:
    # 紧急扩容 20%
    emergency_scale = current_total × 0.2
    scale_out(emergency_scale)
```

**优先级**：延迟触发优先于比例控制，确保快速响应突发流量。

### 配置参数

```yaml
# configs/simulator/baseline_heteroscale.yaml
policy_config:
  # 比例控制参数
  target_tps_per_instance: 100  # 目标单实例TPS
  pd_ratio: 0.33                # P/D比例 (1:3)
  scale_out_threshold: 0.1      # 扩容阈值 (ratio > 1.1)
  scale_in_threshold: 0.1       # 缩容阈值 (ratio < 0.9)
  
  # 延迟触发参数
  tbt_slo: 50.0                 # TBT SLO (毫秒)
  tbt_slo_multiplier: 1.2       # 触发倍数
  emergency_scale_ratio: 0.2    # 紧急扩容幅度 (20%)
  
  # 实例数限制
  min_instances_per_pool: 1
  max_total_instances: 200
```

### 使用方法

```bash
# 运行 HeteroScale 基线
python run.py +experiment=baseline_heteroscale

# 使用自定义 trace
python run.py +experiment=baseline_heteroscale \
  trace.filename=your_trace_name

# 调整策略参数
python run.py +experiment=baseline_heteroscale \
  simulator.policy_config.target_tps_per_instance=120 \
  simulator.policy_config.pd_ratio=0.5
```

## 2. UtilizationBased 策略

### 策略原理

基于显存利用率的简单阈值策略：

```python
# Prompt 池决策
if util_mem_p > 0.8:     # 显存利用率 > 80%
    scale_out_prompt()
elif util_mem_p < 0.3:   # 显存利用率 < 30%
    scale_in_prompt()

# Token 池决策（同理）
if util_mem_t > 0.8:
    scale_out_token()
elif util_mem_t < 0.3:
    scale_in_token()
```

**特点**：
- ✅ 简单直观
- ✅ 直接监控资源使用情况
- ⚠️ 可能对负载变化响应较慢

### 配置参数

```yaml
# configs/simulator/baseline_utilization.yaml
policy_config:
  upper_threshold: 0.8  # 显存利用率高于 80% 时扩容
  lower_threshold: 0.3  # 显存利用率低于 30% 时缩容
  scale_step: 1         # 每次扩/缩容的实例数
  min_instances_per_pool: 1
  max_total_instances: 200
```

### 使用方法

```bash
# 运行基于利用率的基线
python run.py +experiment=baseline_utilization

# 调整阈值
python run.py +experiment=baseline_utilization \
  simulator.policy_config.upper_threshold=0.7 \
  simulator.policy_config.lower_threshold=0.2
```

## 3. QueueBased 策略

### 策略原理

基于队列长度的阈值策略：

```python
# Prompt 队列决策
if p_queue > 1000:      # 队列长度 > 1000 tokens
    scale_out_prompt()
elif p_queue < 100:     # 队列长度 < 100 tokens
    scale_in_prompt()

# Token 队列决策
if d_queue > 5000:      # 队列长度 > 5000 tokens
    scale_out_token()
elif d_queue < 500:     # 队列长度 < 500 tokens
    scale_in_token()
```

**特点**：
- ✅ 直接监控系统负载
- ✅ 对队列堆积敏感
- ⚠️ 需要根据工作负载调整阈值

### 配置参数

```yaml
# configs/simulator/baseline_queue.yaml
policy_config:
  # Prompt 队列阈值 (token 数)
  prompt_queue_upper: 1000
  prompt_queue_lower: 100
  
  # Token 队列阈值 (token 数)
  token_queue_upper: 5000
  token_queue_lower: 500
  
  scale_step: 1
  min_instances_per_pool: 1
  max_total_instances: 200
```

### 使用方法

```bash
# 运行基于队列的基线
python run.py +experiment=baseline_queue

# 调整阈值
python run.py +experiment=baseline_queue \
  simulator.policy_config.prompt_queue_upper=2000 \
  simulator.policy_config.token_queue_upper=8000
```

## 4. 添加自定义基线策略

### 步骤 1：实现策略类

在 `baseline_policies.py` 中继承 `BasePolicy`：

```python
class MyCustomPolicy(BasePolicy):
    def __init__(self, config):
        super().__init__(config)
        # 初始化策略参数
        self.my_param = config.get("my_param", default_value)
    
    def decide(self, state_info, raw_stats, instance_info):
        """
        实现决策逻辑
        
        Args:
            state_info: 状态信息字典
            raw_stats: 原始统计数据
            instance_info: 实例信息
        
        Returns:
            (delta_p, delta_t): prompt/token池的实例数变化
        """
        # 解析输入
        prompt_rate, token_rate, p_queue, d_queue = raw_stats[:4]
        n_p, n_t = instance_info[:2]
        
        # 实现你的决策逻辑
        delta_p = 0
        delta_t = 0
        
        # ... 你的策略逻辑 ...
        
        return delta_p, delta_t
```

### 步骤 2：注册策略

在 `baseline_policies.py` 的 `create_policy` 函数中添加：

```python
policy_map = {
    "heteroscale": HeteroScalePolicy,
    "utilization": UtilizationBasedPolicy,
    "queue": QueueBasedPolicy,
    "mycustom": MyCustomPolicy,  # 添加你的策略
}
```

### 步骤 3：创建配置文件

创建 `configs/simulator/baseline_mycustom.yaml`：

```yaml
algorithm: baseline_mycustom
decision_interval: 2
enabled_features: [...]
policy_config:
  my_param: value
  # ... 其他参数 ...
```

### 步骤 4：创建实验配置

创建 `configs/experiment/baseline_mycustom.yaml`：

```yaml
# @package _global_
defaults:
  - override /simulator: baseline_mycustom
  - override /trace: test_trace
  - _self_

trace_epochs: 1
```

### 步骤 5：运行

```bash
python run.py +experiment=baseline_mycustom
```

## 5. 状态特征说明

基线策略可以访问的状态特征（通过 `raw_stats` 和 `instance_info`）：

### raw_stats（原始统计数据）

索引 | 特征 | 说明 | 单位/格式
-----|------|------|----------
0 | `prompt_rate` | Prompt 生成速率 | prompts/s
1 | `token_rate` | Token 生成速率（TPS） | tokens/s
2 | `p_queue` | Prompt 队列长度 | tokens
3 | `d_queue` | Decoding 队列长度 | tokens
4 | `n_p` | Prompt 实例数 | -
5 | `n_t` | Token 实例数 | -
6 | `avg_prompt_size` | 平均 Prompt 大小 | tokens
7 | `ttft` | Time to First Token | [p50, p90, p99] (ms)
8 | `tbt` | Time Between Tokens | [p50, p90, p99] (ms)
9 | `ins_p_queue` | Prompt 实例队列总长 | requests
10 | `ins_d_queue` | Token 实例队列总长 | requests
11 | `avg_queue_time` | 平均队列等待时间 | s
12 | `avg_nth_token_overhead` | 平均 N-th token 开销 | s
13 | `use_time` | 实例使用时间 | s
14 | `rps` | Requests per Second | req/s

**注意**：`ttft` 和 `tbt` 是列表格式 `[p50, p90, p99]`，包含三个百分位数值。在策略中通常使用 P99 值（索引 2）来判断延迟是否超标。

### instance_info（实例信息）

索引 | 特征 | 说明 | 范围
-----|------|------|------
0 | `n_p` | 活跃 Prompt 实例数 | -
1 | `n_t` | 活跃 Token 实例数 | -
2 | `util_p` | Prompt 池利用率 | [0, 1]
3 | `util_d` | Token 池利用率 | [0, 1]
4 | `util_mem_p` | Prompt 池显存利用率 | [0, 1]
5 | `util_mem_t` | Token 池显存利用率 | [0, 1]

## 6. 评估和对比

### 输出结果

运行基线策略后，会生成以下文件：

```
outputs/baseline_xxx/YYYY-MM-DD/HH-MM-SS/
├── summary.csv           # 汇总结果（TTFT、TBT等指标）
├── reward_xxx.csv        # 奖励曲线（用于评估）
├── detailed/
│   ├── app_0.csv        # 详细调度结果
│   └── app_0_alloc.csv  # 详细分配结果
└── simulator.log         # 模拟器日志
```

### 关键指标

- **TTFT P99**: Time to First Token 99分位数
- **TBT P99**: Time Between Tokens 99分位数
- **平均实例数**: 资源成本指标
- **队列长度**: 系统负载指标
- **奖励曲线**: 综合评估指标

### 对比不同策略

```bash
# 运行多个基线
python run.py +experiment=baseline_heteroscale
python run.py +experiment=baseline_utilization
python run.py +experiment=baseline_queue

# 对比结果（在 summary.csv 中）
```

## 7. 调优建议

### HeteroScale 调优

- **target_tps_per_instance**: 根据实际硬件性能调整
- **pd_ratio**: 根据工作负载特点调整（prompt密集型 → 增大；token密集型 → 减小）
- **scale_out_threshold**: 激进扩容 → 减小；保守扩容 → 增大
- **tbt_slo**: 根据 SLO 要求设置

### UtilizationBased 调优

- **upper_threshold**: 激进扩容 → 降低（如 0.7）；保守扩容 → 提高（如 0.85）
- **lower_threshold**: 激进缩容 → 提高（如 0.4）；保守缩容 → 降低（如 0.2）
- **scale_step**: 增大 → 更快响应，但波动更大

### QueueBased 调优

- **prompt_queue_upper/lower**: 根据队列容忍度调整
- **token_queue_upper/lower**: Token 队列通常比 Prompt 队列更长
- **scale_step**: 队列堆积快时可以增大

## 8. 常见问题

### Q: 基线策略需要训练吗？

A: 不需要。基线策略是规则策略，只需要运行一次 trace 即可。不需要设置 `trace_epochs > 1`。

### Q: 如何快速测试一个基线策略？

A: 使用较短的 trace 和较少的实例数：

```bash
python run.py +experiment=baseline_heteroscale \
  trace.filename=short_trace \
  simulator.max_total_instances=50
```

### Q: 基线策略的奖励是如何计算的？

A: 基线策略复用 RL 框架的 `RLRewardCalculator`，按相同的奖励函数计算奖励（用于评估）。可以通过 `reward_weights` 调整权重。

### Q: 如何添加新的监控指标？

A: 在 `RLStateCollector` 中添加新特征，然后在策略的 `decide` 方法中使用。

## 9. 参考资料

- **HeteroScale 论文**: [相关论文链接]
- **RL 框架文档**: `doc/RL_SCALING_INTEGRATION.md`
- **状态特征配置**: `configs/simulator/baseline_*.yaml`
- **策略实现代码**: `baseline_policies.py`

## 10. 总结

基线策略提供了一个简单、可解释、易于调优的扩缩容方案：

- ✅ 不需要训练，单次运行即可
- ✅ 规则明确，便于理解和调试
- ✅ 可作为 RL 方法的对比基准
- ✅ 可扩展，易于添加自定义策略

建议：
1. 先运行 HeteroScale 作为基准
2. 根据工作负载特点调整参数
3. 与 RL 方法进行对比评估
4. 根据需要实现自定义策略
