# 扩缩容管理器快速开始

## 5 分钟快速上手

### 1. 在配置文件中启用扩缩容管理器

编辑你的配置文件（例如 `config.yaml`），在应用配置中添加：

```yaml
applications:
  - application_id: 0
    # ... 其他配置 ...
    
    # 添加这部分
    scaling_manager:
      scale_up_delay: 10.0           # 扩容延迟（秒）
      drain_check_interval: 1.0      # 排空检查间隔（秒）
```

### 2. 运行仿真

正常运行仿真即可，扩缩容管理器会自动初始化：

```bash
python run.py
```

### 3. 在调度器中触发扩缩容

#### 全流程扩容（推荐）

**一个服务器一个实例**，自动完成服务器 + 实例的扩容：

```python
# 在调度器或其他组件中
scaling_manager = application.scaling_manager

# 全流程扩容：自动创建服务器 + 实例
from model import ModelParallelism

server, instance = scaling_manager.scale_up_full(
    instance_cfg=instance_config,
    parallelism=ModelParallelism(pipeline_parallelism=1, tensor_parallelism=1),
    tag="prompt",  # 或 "token"
    server_sku="A100"  # 可选，默认使用配置中的第一个
)

# server 立即可用
# instance 在 scale_up_delay 秒后自动变为 ACTIVE 状态
```

#### 全流程缩容（推荐）

**一个服务器一个实例**，自动完成实例 + 服务器的缩容：

```python
# 选择要缩容的实例
instance_to_remove = my_instances[0]

# 全流程缩容：自动排空实例 + 移除服务器
scaling_manager.scale_down_full(instance_to_remove)

# 流程：
# 1. 实例立即停止接收新任务（DRAINING）
# 2. 等待实例现有任务完成
# 3. 移除实例
# 4. 自动移除实例所在的服务器（如果该服务器没有其他实例）
```

#### 分步扩缩容（高级用法）

如果需要更细粒度的控制，可以分步操作：

```python
# 分步扩容
server = scaling_manager.scale_up_server(server_cfg)
instance = scaling_manager.scale_up_instance(
    instance_cfg=instance_config,
    processors=server.processors,
    parallelism=parallelism,
    tag="prompt"
)

# 分步缩容
scaling_manager.scale_down_instance(instance)
# 等待实例完全移除后
scaling_manager.scale_down_server(server)
```

#### 自动扩缩容（在调度器中）

```python
class MyScheduler(Scheduler):
    def schedule(self, request):
        # 每 100 个请求检查一次负载
        if self.request_count % 100 == 0:
            self._check_and_scale()
        
        # 正常调度逻辑
        schedulable_instances = self.get_schedulable_instances()
        # ... 调度逻辑 ...
    
    def _check_and_scale(self):
        if self._is_overloaded():
            self._trigger_scale_up()
        elif self._is_underloaded():
            self._trigger_scale_down()
    
    def _is_overloaded(self):
        # 定义过载条件：平均队列长度 > 10
        avg_queue = sum(len(i.pending_queue) for i in self.instances) / len(self.instances)
        return avg_queue > 10
    
    def _trigger_scale_up(self):
        """全流程扩容：服务器 + 实例"""
        from model import ModelParallelism
        
        scaling_manager = self.application.scaling_manager
        instance_cfg = self._get_instance_config()
        parallelism = ModelParallelism(pipeline_parallelism=1, tensor_parallelism=1)
        
        # 一行代码完成扩容
        server, instance = scaling_manager.scale_up_full(
            instance_cfg=instance_cfg,
            parallelism=parallelism,
            tag="prompt"
        )
        print(f"Scaled up: server {server.server_id} + instance {instance.instance_id}")
    
    def _trigger_scale_down(self):
        """全流程缩容：实例 + 服务器"""
        if len(self.instances) > 1:  # 保持至少一个实例
            least_loaded = min(self.instances, key=lambda i: len(i.pending_queue))
            
            # 一行代码完成缩容
            self.application.scaling_manager.scale_down_full(least_loaded)
            print(f"Scaling down: instance {least_loaded.instance_id}")
```

### 4. 配置自动获取

扩缩容管理器会自动从启动状态配置中获取实例配置和并行度：

```python
# 在调度器中自动使用配置
def _scale_up_prompt_instance(self):
    # 自动从 start_state_manager 获取配置
    instance_cfg = self._get_instance_config("prompt")
    parallelism = self._get_parallelism("prompt")
    
    # 使用配置进行扩容
    server, instance = self.scaling_manager.scale_up_full(
        instance_cfg=instance_cfg,
        parallelism=parallelism,
        tag="prompt"
    )
```

**配置来源：** `start_state.py` 中的 `StartStateManager` 会保存启动配置，包括：
- Prompt 实例配置和并行度
- Token 实例配置和并行度
- 实例类型、批处理大小等参数

### 5. 查看日志

扩缩容操作会记录在日志文件中：

```
scaling/{application_id}
```

日志格式：
```
time,action,target,status
10.5,scale_up_start,instance_5,scaling_up
20.5,scale_up_complete,instance_5,active
100.0,scale_down_start,instance_3,draining
125.0,scale_down_complete,instance_3,scaling_down
```

## 关键概念

### 实例状态

| 状态 | 说明 | 可调度 |
|------|------|--------|
| ACTIVE | 正常运行 | ✅ |
| SCALING_UP | 正在启动 | ❌ |
| DRAINING | 排空中 | ❌ |
| SCALING_DOWN | 已缩容 | ❌ |

### 自动过滤

调度器会自动过滤掉不可调度的实例：

```python
# 以前：直接使用所有实例
instance = min(self.instances, key=lambda i: i.load)

# 现在：自动过滤
schedulable = self.get_schedulable_instances()
instance = min(schedulable, key=lambda i: i.load)
```

## 常见使用场景

### 场景 1：负载感知扩容

```python
def schedule(self, request):
    # 每 100 个请求检查一次
    if self.request_count % 100 == 0:
        if self._high_load():
            self._scale_up()
```

### 场景 2：时间触发缩容

```python
# 在仿真事件中定期检查
schedule_event(600.0,  # 每 10 分钟
    lambda: self._check_and_scale_down())
```

### 场景 3：阈值触发

```python
# 队列长度超过阈值
if avg_queue_length > 50:
    scaling_manager.scale_up_instance(...)

# 利用率低于阈值
if utilization < 0.2:
    scaling_manager.scale_down_instance(...)
```

## 完整示例

参考以下文件：

- **详细指南**：`SCALING_MANAGER_GUIDE.md`
- **配置示例**：`example_scaling_config.yaml`
- **代码示例**：`example_autoscaling_scheduler.py`
- **实现总结**：`SCALING_IMPLEMENTATION_SUMMARY.md`

## 故障排除

### 问题：实例没有变为 ACTIVE

**原因**：需要等待 `scale_up_delay` 秒

**解决**：检查配置中的 `scale_up_delay` 值

### 问题：缩容实例卡住

**原因**：实例还有待处理的任务

**解决**：检查实例的 `pending_queue`、`batch`、`blocked_queue`

### 问题：调度器找不到可用实例

**原因**：所有实例都在扩缩容中

**解决**：
1. 避免同时缩容所有实例
2. 保持最小实例数
3. 检查扩容是否成功

## 下一步

- 阅读 `SCALING_MANAGER_GUIDE.md` 了解详细功能
- 查看 `example_autoscaling_scheduler.py` 学习自动扩缩容
- 实现自己的扩缩容策略

## 支持

如果遇到问题：
1. 检查日志文件：`scaling/{application_id}`
2. 查看实例状态：`scaling_manager.get_status_summary()`
3. 参考文档和示例代码

