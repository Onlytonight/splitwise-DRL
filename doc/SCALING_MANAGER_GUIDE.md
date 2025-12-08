# 扩缩容管理器使用指南

## 概述

扩缩容管理器（ScalingManager）提供了对集群中服务器和实例进行动态扩缩容的能力。它支持：

- ✅ 实例扩容（带启动延迟）
- ✅ 实例缩容（优雅排空任务）
- ✅ 服务器扩容
- ✅ 服务器缩容
- ✅ 实例状态管理（防止向缩容中的实例调度任务）
- ✅ 与现有调度器无缝集成

## 核心组件

### 1. ScalingManager 类

位于 `scaling_manager.py`，负责管理扩缩容操作。

**主要方法：**

- `scale_up_full()` - **全流程扩容**（服务器 + 实例，推荐使用）
- `scale_down_full()` - **全流程缩容**（实例 + 服务器，推荐使用）
- `scale_up_instance()` - 扩容实例（内部方法）
- `scale_down_instance()` - 缩容实例（内部方法）
- `scale_up_server()` - 扩容服务器（内部方法）
- `scale_down_server()` - 缩容服务器（内部方法）
- `can_schedule_to_instance()` - 检查实例是否可调度
- `get_active_instances()` - 获取活跃实例列表

### 2. 实例状态（InstanceStatus）

- `ACTIVE` - 正常运行，可以接收新任务
- `SCALING_UP` - 正在扩容启动中，不可接收任务
- `DRAINING` - 排空中，不接收新任务，等待现有任务完成
- `SCALING_DOWN` - 准备缩容，已排空

### 3. 服务器状态（ServerStatus）

- `ACTIVE` - 正常运行
- `SCALING_UP` - 正在启动中
- `SCALING_DOWN` - 准备关闭

## 使用方法

### 1. 在配置文件中启用扩缩容管理器

在应用配置中添加 `scaling_manager` 配置：

```yaml
applications:
  - application_id: 0
    model_architecture: LLaMA
    model_size: 70B
    scheduler: AdaptiveMixedPoolScheduler
    allocator: StaticAllocator
    debug: true
    
    # 扩缩容管理器配置
    scaling_manager:
      scale_up_delay: 10.0  # 扩容启动延迟（秒）
      drain_check_interval: 1.0  # 排空检查间隔（秒）
```

### 2. 在代码中使用扩缩容管理器

#### ⭐ 全流程扩容（推荐）

**一个服务器一个实例**，一行代码完成扩容：

```python
# 获取扩缩容管理器
scaling_manager = application.scaling_manager

# 全流程扩容：自动创建服务器 + 实例
from model import ModelParallelism

instance_cfg = ...  # 从配置获取
parallelism = ModelParallelism(pipeline_parallelism=1, tensor_parallelism=1)

# 一行代码完成扩容
server, instance = scaling_manager.scale_up_full(
    instance_cfg=instance_cfg,
    parallelism=parallelism,
    tag="prompt",  # 或 "token"
    server_sku="A100"  # 可选，默认使用配置中的第一个
)

# 流程：
# 1. 自动创建新服务器
# 2. 在新服务器上创建实例（使用所有处理器）
# 3. 实例在 scale_up_delay 秒后变为 ACTIVE
```

#### ⭐ 全流程缩容（推荐）

**一个服务器一个实例**，一行代码完成缩容：

```python
# 选择要缩容的实例
instance_to_remove = my_instances[0]

# 全流程缩容：自动排空实例 + 移除服务器
scaling_manager.scale_down_full(instance_to_remove)

# 流程：
# 1. 实例立即标记为 DRAINING，停止接收新任务
# 2. 等待所有现有任务完成
# 3. 移除实例
# 4. 自动移除实例所在的服务器（如果该服务器没有其他实例）
```

#### 分步扩缩容（高级用法）

如果需要更细粒度的控制，可以使用内部方法：

```python
# 分步扩容
server_cfg = cluster.get_server_config("A100")
server = scaling_manager.scale_up_server(server_cfg)

instance = scaling_manager.scale_up_instance(
    instance_cfg=instance_cfg,
    processors=server.processors,
    parallelism=parallelism,
    tag="prompt"
)

# 分步缩容
scaling_manager.scale_down_instance(instance)
# 等待实例完全移除后
if len(server.instances) == 0:
    scaling_manager.scale_down_server(server)
```

### 3. 调度器自动集成

所有调度器都会自动过滤掉不可调度的实例：

```python
# 调度器内部会自动调用
schedulable_instances = self.get_schedulable_instances()

# 只会在 ACTIVE 状态的实例上调度任务
```

### 4. 在调度器中触发扩缩容

调度器可以根据负载情况触发扩缩容：

```python
class AdaptiveScheduler(Scheduler):
    def schedule(self, request):
        # 检查负载
        if self.is_overloaded():
            # 触发扩容
            self.trigger_scale_up()
        elif self.is_underloaded():
            # 触发缩容
            self.trigger_scale_down()
        
        # 正常调度
        ...
    
    def trigger_scale_up(self):
        """触发实例扩容"""
        if not hasattr(self.application, 'scaling_manager'):
            return
        
        scaling_manager = self.application.scaling_manager
        
        # 1. 选择或创建服务器
        # 如果有空闲服务器，直接使用
        available_server = self._find_available_server()
        if available_server is None:
            # 需要扩容新服务器
            server_cfg = self.application.cluster.get_server_config("A100")
            available_server = scaling_manager.scale_up_server(server_cfg)
        
        # 2. 在服务器上创建实例
        from model import ModelParallelism
        instance_cfg = ...
        parallelism = ModelParallelism(pipeline_parallelism=1, tensor_parallelism=1)
        
        new_instance = scaling_manager.scale_up_instance(
            instance_cfg=instance_cfg,
            processors=available_server.processors,
            parallelism=parallelism,
            tag="prompt"
        )
        
        print(f"Triggered scale-up: new instance {new_instance.instance_id}")
    
    def trigger_scale_down(self):
        """触发实例缩容"""
        if not hasattr(self.application, 'scaling_manager'):
            return
        
        scaling_manager = self.application.scaling_manager
        
        # 选择负载最低的实例进行缩容
        if len(self.instances) > 1:  # 保留至少一个实例
            least_loaded = min(self.instances, 
                             key=lambda i: len(i.pending_queue))
            scaling_manager.scale_down_instance(least_loaded)
            print(f"Triggered scale-down: instance {least_loaded.instance_id}")
```

## 扩缩容流程

### 全流程扩容（scale_up_full）

**一个服务器一个实例**

```
[调用 scale_up_full()]
    ↓
[自动创建新服务器] → [服务器: ACTIVE]
    ↓
[在新服务器上创建实例] → [实例: SCALING_UP]
    ↓
[等待 scale_up_delay 秒]
    ↓
[实例: ACTIVE] → 可以接收任务
```

### 全流程缩容（scale_down_full）

**一个服务器一个实例**

```
[调用 scale_down_full()]
    ↓
[记录实例所在服务器]
    ↓
[实例: DRAINING] → 停止接收新任务
    ↓
[等待现有任务完成] ← 定期检查（每 drain_check_interval）
    ↓
[所有任务完成]
    ↓
[从应用和调度器移除实例]
    ↓
[实例: SCALING_DOWN]
    ↓
[检查服务器是否还有其他实例]
    ↓
[如果没有] → [自动移除服务器]
```

### 分步扩容流程（高级用法）

```
[触发扩容]
    ↓
[选择/创建服务器] ← 手动调用 scale_up_server()
    ↓
[创建实例] → [状态: SCALING_UP] ← 手动调用 scale_up_instance()
    ↓
[等待 scale_up_delay 秒]
    ↓
[状态: ACTIVE] → 可以接收任务
```

### 分步缩容流程（高级用法）

```
[触发缩容]
    ↓
[检查实例状态] ← 手动调用 scale_down_instance()
    ↓
[状态: DRAINING] → 停止接收新任务
    ↓
[等待现有任务完成] ← 定期检查
    ↓
[所有任务完成]
    ↓
[从应用和调度器移除]
    ↓
[状态: SCALING_DOWN]
    ↓
[手动检查并移除服务器] ← 手动调用 scale_down_server()
```

## 实例状态检查

在调度时，扩缩容管理器会自动过滤实例：

```python
# 在 Instance 类中
def is_active_for_scheduling(self):
    if hasattr(self.application, 'scaling_manager'):
        return self.application.scaling_manager.can_schedule_to_instance(self)
    return True

# 在 Scheduler 类中
def get_schedulable_instances(self, instances=None):
    if instances is None:
        instances = self.instances
    
    if hasattr(self.application, 'scaling_manager'):
        return self.application.scaling_manager.get_active_instances(instances)
    
    return [inst for inst in instances if inst.is_active_for_scheduling()]
```

## 监控和日志

扩缩容管理器会记录所有操作：

```
time,action,target,status
10.5,scale_up_start,instance_5,scaling_up
20.5,scale_up_complete,instance_5,active
100.0,scale_down_start,instance_3,draining
125.0,scale_down_complete,instance_3,scaling_down
```

日志位置：`scaling/{application_id}`

## 注意事项

1. **扩容延迟**：新实例需要 `scale_up_delay` 秒才能变为 ACTIVE
2. **缩容等待**：实例缩容会等待所有任务完成，可能需要较长时间
3. **最小实例数**：建议在调度器中保持至少一个实例
4. **服务器缩容**：只能缩容没有实例的服务器
5. **配置保存**：集群配置会在初始化时保存，用于后续扩容

## 示例场景

### 场景1：负载增加时自动扩容

```python
# 在调度器中检测到队列过长
if avg_queue_length > threshold:
    scaling_manager.scale_up_instance(...)
```

### 场景2：负载降低时自动缩容

```python
# 检测到实例利用率低
if instance.utilization < 0.2:
    scaling_manager.scale_down_instance(instance)
```

### 场景3：预测性扩缩容

```python
# 根据历史负载预测
if predicted_load > current_capacity:
    # 提前扩容，考虑 scale_up_delay
    scaling_manager.scale_up_instance(...)
```

## 配置示例

完整的配置文件示例：

```yaml
cluster:
  servers:
    - sku: A100
      count: 4
  interconnects:
    - topology: p2p
  power_budget: 100000

applications:
  - application_id: 0
    model_architecture: LLaMA
    model_size: 70B
    scheduler: AdaptiveMixedPoolScheduler
    allocator: StaticAllocator
    
    # 启用扩缩容管理器
    scaling_manager:
      scale_up_delay: 10.0
      drain_check_interval: 1.0
    
    debug: true

start_state:
  state_type: splitwise
  application_id: 0
  split_type: homogeneous
  prompt:
    num_instances: 2
    pipeline_parallelism: 1
    tensor_parallelism: 1
  token:
    num_instances: 2
    pipeline_parallelism: 1
    tensor_parallelism: 1
```

## API 参考

### ScalingManager

#### `__init__(application, cluster, scale_up_delay=10.0, drain_check_interval=1.0, debug=False)`

初始化扩缩容管理器。

**参数：**
- `application`: 应用实例
- `cluster`: 集群实例
- `scale_up_delay`: 扩容启动延迟（秒）
- `drain_check_interval`: 排空检查间隔（秒）
- `debug`: 是否启用调试日志

#### `scale_up_full(instance_cfg, parallelism, tag=None, server_sku=None)` ⭐

**推荐使用**。全流程扩容：自动创建服务器 + 实例。

**参数：**
- `instance_cfg`: 实例配置
- `parallelism`: 模型并行配置
- `tag`: 实例标签（"prompt" 或 "token"）
- `server_sku`: 服务器 SKU 名称（可选，默认使用配置中的第一个）

**返回：** `(server, instance)` 元组

**约束：** 一个服务器一个实例

#### `scale_down_full(instance)` ⭐

**推荐使用**。全流程缩容：自动排空实例 + 移除服务器。

**参数：**
- `instance`: 要缩容的实例

**流程：**
1. 实例标记为 DRAINING
2. 等待任务完成
3. 移除实例
4. 自动移除服务器（如果该服务器没有其他实例）

**约束：** 一个服务器一个实例

#### `scale_up_instance(instance_cfg, processors, parallelism, tag=None)`

扩容实例（内部方法，通常应使用 `scale_up_full()`）。

**返回：** 新创建的实例（SCALING_UP 状态）

#### `scale_down_instance(instance)`

缩容实例（内部方法，通常应使用 `scale_down_full()`）。

#### `scale_up_server(server_cfg)`

扩容服务器（内部方法，通常应使用 `scale_up_full()`）。

**返回：** 新创建的服务器

#### `scale_down_server(server)`

缩容服务器（内部方法，需要先排空所有实例）。

#### `can_schedule_to_instance(instance)`

检查实例是否可以接收新任务。

**返回：** `bool`

#### `get_active_instances(instances)`

从实例列表中过滤出活跃实例。

**返回：** `list`

#### `get_status_summary()`

获取扩缩容状态摘要。

**返回：** `dict` 包含各类实例和服务器的数量

## 扩展示例

如果需要更复杂的扩缩容策略，可以继承 `ScalingManager` 或在调度器中实现：

```python
class PredictiveScalingManager(ScalingManager):
    def __init__(self, *args, predictor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor = predictor
    
    def check_and_scale(self):
        """定期检查并执行预测性扩缩容"""
        predicted_load = self.predictor.predict_next_interval()
        current_capacity = self.calculate_capacity()
        
        if predicted_load > current_capacity * 1.2:
            self.trigger_scale_up()
        elif predicted_load < current_capacity * 0.5:
            self.trigger_scale_down()
```

## 总结

扩缩容管理器提供了一个完整的、与调度器解耦的扩缩容解决方案。它支持：

1. ✅ 动态扩缩容实例和服务器
2. ✅ 状态管理和调度保护
3. ✅ 优雅的任务排空
4. ✅ 配置驱动的扩容
5. ✅ 与现有调度器无缝集成

通过调度器触发扩缩容，系统可以根据实际负载动态调整资源，提高资源利用率。

