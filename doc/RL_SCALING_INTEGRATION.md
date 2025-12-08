# RL 扩缩容管理器集成文档

## 概述

将强化学习（RL）的扩缩容决策与现有的扩缩容管理器集成，实现自动化的、智能的资源管理。

## 主要修改

### 1. `RL/action.py` - 动作执行器

#### 修改内容

**之前：** 直接调用 `cluster.add_instances()` 和 `cluster.remove_instances()`

**现在：** 使用扩缩容管理器的 `scale_up_full()` 和 `scale_down_full()` 方法

#### 关键变化

```python
# 初始化
class RLActionExecutor:
    def __init__(self, application, config):
        """
        :param application: 应用对象（包含 scaling_manager 和 scheduler）
        """
        self.application = application
        self.scaling_manager = application.scaling_manager
        self.scheduler = application.scheduler
```

#### 扩容实现

```python
# 使用全流程扩容：自动创建服务器 + 实例
instance_cfg = self._get_instance_config(tag)
parallelism = self._get_parallelism(tag)

server, instance = self.scaling_manager.scale_up_full(
    instance_cfg=instance_cfg,
    parallelism=parallelism,
    tag=tag,
    server_sku=None  # 使用默认 SKU
)
```

**优势：**
- ✅ 自动创建服务器
- ✅ 自动配置实例
- ✅ 一个服务器一个实例
- ✅ 使用配置文件中的参数

#### 缩容实现

```python
# 选择负载最低的实例
if tag == "prompt":
    least_loaded = min(active_instances, key=lambda i: i.sched_pending_tokens)
else:  # token
    least_loaded = min(active_instances, key=lambda i: i.sched_memory)

# 使用全流程缩容：自动排空实例 + 移除服务器
self.scaling_manager.scale_down_full(least_loaded)
```

**优势：**
- ✅ 自动选择最佳缩容目标
- ✅ 优雅排空任务
- ✅ 自动移除空闲服务器

#### 配置获取

新增两个辅助方法，从启动状态配置自动获取实例参数：

```python
def _get_instance_config(self, tag):
    """从启动状态配置获取实例配置"""
    if hasattr(self.application, 'start_state_manager'):
        return self.application.start_state_manager.get_instance_config(tag)
    return default_config

def _get_parallelism(self, tag):
    """从启动状态配置获取并行度"""
    if hasattr(self.application, 'start_state_manager'):
        return self.application.start_state_manager.get_parallelism(tag)
    return default_parallelism
```

### 2. `simulator.py` - RL 仿真器

#### 修改内容

```python
# 获取第一个应用（假设只有一个应用）
self.application = list(applications.values())[0]

self.action_executor = RLActionExecutor(
    application=self.application,  # 传入应用而非集群
    config=rl_config
)
```

**关键变化：**
- 传入 `application` 对象而非 `cluster` 对象
- 通过 `application` 访问扩缩容管理器、调度器和配置

### 3. `RL/state.py` - 状态收集器

#### 修改内容

在状态收集时，只统计活跃实例（排除扩缩容中的实例）：

```python
def get_instance_feature(self):
    # 获取活跃实例（排除扩缩容中的实例）
    if hasattr(self.applications[0], 'scaling_manager'):
        active_instances = self.applications[0].scaling_manager.get_active_instances(
            scheduler.instances
        )
    else:
        active_instances = scheduler.instances
    
    # 只统计活跃实例
    for instance in active_instances:
        # ... 统计逻辑 ...
```

**修改的方法：**
- `get_instance_feature()` - 只统计活跃实例
- `compute_util()` - 只计算活跃实例的利用率
- `get_avg_utilization()` - 使用过滤后的实例

**优势：**
- ✅ 避免统计正在启动的实例
- ✅ 避免统计正在排空的实例
- ✅ 更准确的负载和利用率计算

## 集成流程

### 1. 初始化

```
[TraceRLSimulator 初始化]
    ↓
[获取 application 对象]
    ↓
[创建 RLActionExecutor(application)]
    ↓
[创建 RLStateCollector(applications)]
    ↓
[创建 RLRewardCalculator()]
```

### 2. RL 决策循环

```
[每隔 decision_interval 秒]
    ↓
[RLStateCollector 收集状态]
    ↓ (只统计活跃实例)
[PPO Agent 推理动作]
    ↓
[RLActionExecutor 执行动作]
    ↓ (使用扩缩容管理器)
[扩容：scale_up_full()]
[缩容：scale_down_full()]
    ↓
[RLRewardCalculator 计算奖励]
    ↓
[PPO Agent 更新策略]
```

### 3. 扩容流程

```
[RL Agent 决定扩容]
    ↓
[RLActionExecutor._handle_scaling(delta=+N, tag="prompt")]
    ↓
[循环 N 次]
    ↓
[获取实例配置和并行度]
    ↓
[scaling_manager.scale_up_full()]
    ↓
[自动创建服务器]
    ↓
[在服务器上创建实例]
    ↓
[实例延迟后变为 ACTIVE]
```

### 4. 缩容流程

```
[RL Agent 决定缩容]
    ↓
[RLActionExecutor._handle_scaling(delta=-N, tag="token")]
    ↓
[循环 N 次]
    ↓
[选择负载最低的活跃实例]
    ↓
[scaling_manager.scale_down_full()]
    ↓
[实例停止接收新任务 (DRAINING)]
    ↓
[等待现有任务完成]
    ↓
[移除实例]
    ↓
[自动移除空闲服务器]
```

## 优势

### 1. 配置一致性

- ✅ 扩容的实例与初始实例使用相同配置
- ✅ 配置来自启动状态配置文件
- ✅ 避免硬编码

### 2. 自动化

- ✅ 自动创建和移除服务器
- ✅ 自动优雅排空任务
- ✅ 自动状态管理

### 3. 准确性

- ✅ 状态收集只统计活跃实例
- ✅ 避免计入正在启动/排空的实例
- ✅ 更准确的负载评估

### 4. 健壮性

- ✅ 遵守最小/最大实例数限制
- ✅ 检查集群容量上限
- ✅ 错误处理和日志记录

### 5. 一致性

- ✅ 与手动扩缩容使用相同的管理器
- ✅ 统一的扩缩容流程
- ✅ 统一的状态管理

## 使用方法

### 1. 确保应用有扩缩容管理器

在配置文件中启用：

```yaml
applications:
  - application_id: 0
    scaling_manager:
      scale_up_delay: 10.0
      drain_check_interval: 1.0
```

### 2. 配置 RL 参数

```python
rl_config = {
    "action_scale_step": 5,        # 扩缩容步长
    "min_instances_per_pool": 1,   # 最小实例数
    "max_total_instances": 100,    # 最大总实例数
    # ... 其他参数 ...
}
```

### 3. 运行 RL 仿真

```python
sim = TraceRLSimulator(
    trace=trace,
    cluster=cluster,
    applications=applications,
    router=router,
    arbiter=arbiter,
    end_time=end_time
)
sim.run()
```

## 配置参数

### RLActionExecutor

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `action_scale_step` | 扩缩容步长（PPO 输出 1.0 对应的实例数） | 5 |
| `action_mig_step` | 迁移步长 | 3 |
| `min_instances_per_pool` | 每个池的最小实例数 | 1 |
| `max_total_instances` | 集群最大总实例数 | 100 |

### ScalingManager

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `scale_up_delay` | 扩容启动延迟（秒） | 10.0 |
| `drain_check_interval` | 排空检查间隔（秒） | 1.0 |

## 监控和调试

### 日志输出

#### RLActionExecutor 日志

```
[Action] Added prompt instance 5 on server 10
[Action] Removed token instance 3
[Action Blocked] Max cluster size reached (100)
[Action Blocked] Cannot remove prompt, min limit reached
```

#### ScalingManager 日志

```
[ScalingManager] Full scale-up: server 10 + instance 5
[ScalingManager] Instance 5 scaled up at 120.50
[ScalingManager] Instance 3 draining started at 250.00
[ScalingManager] Auto scale-down server 8 after instance removal
```

### 状态监控

```python
# 获取扩缩容状态摘要
summary = scaling_manager.get_status_summary()
print(summary)
# {
#     'active_instances': 10,
#     'scaling_up_instances': 2,
#     'draining_instances': 1,
#     'total_servers': 10,
#     'total_instances': 13
# }
```

### RL 训练监控

```python
# 每 100 步输出一次
if self.decision_step % 100 == 0:
    logging.info(f"Step: {self.decision_step} | "
                f"Reward: {reward:.4f} | "
                f"Active Instances: {n_p + n_t + n_m}")
```

## 注意事项

### 1. 扩容延迟

- 新实例需要 `scale_up_delay` 秒才能变为 ACTIVE
- RL Agent 需要考虑这个延迟
- 可能需要预测性扩容

### 2. 缩容等待

- 实例缩容需要等待任务完成
- 可能需要较长时间
- 不会阻塞仿真运行

### 3. 状态一致性

- 状态收集只统计活跃实例
- 确保 RL Agent 看到的是实际可用容量
- 避免基于正在启动的实例做决策

### 4. 最小实例数

- 确保每个池至少有 `min_instances_per_pool` 个实例
- 防止完全缩容导致服务中断

### 5. 最大实例数

- 遵守 `max_total_instances` 限制
- 防止无限扩容

## 测试建议

### 1. 单元测试

```python
def test_rl_action_executor_scale_up():
    """测试 RL 动作执行器扩容"""
    executor = RLActionExecutor(application, config)
    
    # 模拟扩容动作
    action = [0.5, 0.0, 0.0]  # 扩容 prompt
    executor.execute(action)
    
    # 验证实例创建
    assert len(scheduler.prompt_instances) > initial_count
```

### 2. 集成测试

```python
def test_rl_simulator_full_cycle():
    """测试完整的 RL 仿真周期"""
    sim = TraceRLSimulator(...)
    sim.run()
    
    # 验证决策执行
    assert sim.decision_step > 0
    assert len(sim.agent.buffer.rewards) > 0
```

### 3. 性能测试

- 测试在高负载下的扩容响应
- 测试在低负载下的缩容效果
- 测试 RL Agent 的收敛性

## 故障排除

### 问题：扩容失败

**原因：** 配置缺失或集群容量达到上限

**解决：**
1. 检查 `start_state_manager` 是否存在
2. 检查集群配置是否正确
3. 检查是否达到 `max_total_instances`

### 问题：缩容卡住

**原因：** 实例还有待处理任务

**解决：**
1. 检查实例的任务队列
2. 增加 `drain_check_interval`
3. 查看扩缩容管理器日志

### 问题：状态统计不准确

**原因：** 包含了扩缩容中的实例

**解决：**
- 确保使用 `get_active_instances()` 过滤实例
- 检查 `scaling_manager` 是否正确初始化

## 总结

### ✅ 已完成

- ✅ 集成扩缩容管理器到 RL 动作执行器
- ✅ 使用全流程扩缩容方法
- ✅ 自动配置获取
- ✅ 状态收集过滤活跃实例
- ✅ 完整的日志和错误处理

### 🎯 优势

- 配置一致性
- 自动化程度高
- 准确的状态评估
- 健壮的错误处理
- 与手动扩缩容统一

### 📈 后续改进

1. **预测性扩容** - 考虑启动延迟，提前扩容
2. **智能缩容** - 基于任务预测选择缩容时机
3. **多目标优化** - 平衡性能、成本和稳定性
4. **自适应步长** - 根据负载动态调整扩缩容步长

---

**状态：已完成 ✅**  
**测试：通过 Linter 检查 ✅**  
**集成：完全兼容现有系统 ✅**

