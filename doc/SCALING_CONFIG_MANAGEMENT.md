# 扩缩容配置管理

## 概述

扩缩容管理器现在与启动状态配置集成，自动获取实例配置和并行度，无需在调度器中硬编码配置。

## StartStateManager

`StartStateManager` 类负责管理启动状态配置，提供统一的配置访问接口。

### 初始化

`StartStateManager` 在启动时自动创建并保存到应用中：

```python
# 在 start_state.py 的 load_start_state 函数中
start_state_manager = StartStateManager(start_state_cfg)

# 保存到应用
for app in applications.values():
    app.start_state_manager = start_state_manager
```

### 配置来源

配置来自启动状态配置文件：

```yaml
start_state:
  state_type: splitwise
  application_id: 0
  split_type: homogeneous
  
  # Prompt 实例配置
  prompt:
    num_instances: 2
    pipeline_parallelism: 1
    tensor_parallelism: 1
    instance_type: Splitwise
    max_batch_size: 64
    max_batch_tokens: 4096
    max_preemptions: 3
  
  # Token 实例配置
  token:
    num_instances: 2
    pipeline_parallelism: 1
    tensor_parallelism: 1
    instance_type: Splitwise
    max_batch_size: 64
    max_batch_tokens: 4096
    max_preemptions: 3
```

## API 方法

### `get_instance_config(tag=None)`

获取实例配置。

**参数：**
- `tag`: 实例标签（"prompt" 或 "token"）
- 如果为 `None`，返回默认配置（通常是 prompt 配置）

**返回：** 实例配置对象

**示例：**

```python
# 获取 prompt 实例配置
prompt_cfg = app.start_state_manager.get_instance_config("prompt")

# 获取 token 实例配置
token_cfg = app.start_state_manager.get_instance_config("token")

# 使用配置
print(f"Max batch size: {prompt_cfg.max_batch_size}")
print(f"Max batch tokens: {prompt_cfg.max_batch_tokens}")
```

### `get_parallelism(tag=None)`

获取并行度配置。

**参数：**
- `tag`: 实例标签（"prompt" 或 "token"）
- 如果为 `None`，返回默认并行度

**返回：** `ModelParallelism` 对象

**示例：**

```python
# 获取 prompt 并行度
prompt_parallelism = app.start_state_manager.get_parallelism("prompt")

# 获取 token 并行度
token_parallelism = app.start_state_manager.get_parallelism("token")

# 使用并行度
print(f"Pipeline: {prompt_parallelism.pipeline_parallelism}")
print(f"Tensor: {prompt_parallelism.tensor_parallelism}")
```

## 在调度器中使用

### 基本用法

```python
class MyScheduler(KVScheduler):
    def _scale_up_prompt_instance(self):
        # 从配置获取
        instance_cfg = self._get_instance_config("prompt")
        parallelism = self._get_parallelism("prompt")
        
        # 使用配置进行扩容
        server, instance = self.application.scaling_manager.scale_up_full(
            instance_cfg=instance_cfg,
            parallelism=parallelism,
            tag="prompt"
        )
    
    def _get_instance_config(self, tag):
        """从启动状态配置获取实例配置"""
        if hasattr(self.application, 'start_state_manager') and \
           self.application.start_state_manager is not None:
            return self.application.start_state_manager.get_instance_config(tag)
        
        # 回退到默认配置
        return self._get_default_config()
    
    def _get_parallelism(self, tag):
        """从启动状态配置获取并行度"""
        if hasattr(self.application, 'start_state_manager') and \
           self.application.start_state_manager is not None:
            return self.application.start_state_manager.get_parallelism(tag)
        
        # 回退到默认并行度
        from model import ModelParallelism
        return ModelParallelism(pipeline_parallelism=1, tensor_parallelism=1)
```

### 完整示例

参见 `example_autoscaling_scheduler.py`：

```python
class AutoScalingScheduler(KVScheduler):
    def _scale_up_prompt_instance(self):
        """扩容 Prompt 实例（使用配置）"""
        # 自动从配置获取
        instance_cfg = self._get_instance_config("prompt")
        parallelism = self._get_parallelism("prompt")
        
        if instance_cfg is None or parallelism is None:
            print(f"[AutoScaling] No configuration found")
            return
        
        # 扩容
        try:
            server, instance = self.application.scaling_manager.scale_up_full(
                instance_cfg=instance_cfg,
                parallelism=parallelism,
                tag="prompt"
            )
            print(f"[AutoScaling] Created server {server.server_id} "
                  f"with instance {instance.instance_id}")
        except Exception as e:
            print(f"[AutoScaling] Failed to scale up: {e}")
    
    def _scale_up_token_instance(self):
        """扩容 Token 实例（使用配置）"""
        # 自动从配置获取
        instance_cfg = self._get_instance_config("token")
        parallelism = self._get_parallelism("token")
        
        if instance_cfg is None or parallelism is None:
            print(f"[AutoScaling] No configuration found")
            return
        
        # 扩容
        try:
            server, instance = self.application.scaling_manager.scale_up_full(
                instance_cfg=instance_cfg,
                parallelism=parallelism,
                tag="token"
            )
            print(f"[AutoScaling] Created server {server.server_id} "
                  f"with instance {instance.instance_id}")
        except Exception as e:
            print(f"[AutoScaling] Failed to scale up: {e}")
```

## 配置类型支持

### Splitwise 配置

支持独立的 prompt 和 token 配置：

```python
# 获取 prompt 配置
prompt_cfg = manager.get_instance_config("prompt")
prompt_parallelism = manager.get_parallelism("prompt")

# 获取 token 配置
token_cfg = manager.get_instance_config("token")
token_parallelism = manager.get_parallelism("token")
```

### ORCA/Baseline 配置

使用统一的实例配置：

```python
# 获取统一配置
instance_cfg = manager.get_instance_config()
parallelism = manager.get_parallelism()
```

## 优势

### 1. 配置集中管理

- ✅ 所有配置在启动配置文件中定义
- ✅ 避免在调度器中硬编码
- ✅ 易于修改和维护

### 2. 一致性保证

- ✅ 新实例与初始实例使用相同配置
- ✅ 避免配置不一致导致的问题
- ✅ 统一的配置来源

### 3. 灵活性

- ✅ 支持多种启动状态类型
- ✅ 自动适配 prompt/token 分离或统一配置
- ✅ 提供默认配置回退机制

### 4. 简化调度器

- ✅ 无需在调度器中创建配置对象
- ✅ 减少代码重复
- ✅ 更易于扩展

## 配置流程

```
[启动配置文件 (YAML)]
    ↓
[load_start_state()]
    ↓
[创建 StartStateManager]
    ↓
[保存到 Application.start_state_manager]
    ↓
[调度器调用 get_instance_config()/get_parallelism()]
    ↓
[获取配置用于扩容]
```

## 错误处理

### 配置缺失

如果 `start_state_manager` 不存在或配置缺失：

```python
def _get_instance_config(self, tag):
    if hasattr(self.application, 'start_state_manager') and \
       self.application.start_state_manager is not None:
        return self.application.start_state_manager.get_instance_config(tag)
    
    # 回退：返回默认配置或 None
    print(f"[Warning] No start_state_manager found, using default config")
    return self._get_default_config()
```

### 无效标签

如果标签无效，返回默认配置（通常是 prompt 配置）：

```python
# tag=None 或无效标签时返回默认配置
instance_cfg = manager.get_instance_config(None)  # 返回 prompt 配置
```

## 最佳实践

### 1. 始终检查配置存在性

```python
instance_cfg = self._get_instance_config(tag)
if instance_cfg is None:
    print(f"[Error] No configuration found for {tag}")
    return
```

### 2. 使用辅助方法

创建辅助方法封装配置访问：

```python
def _get_instance_config(self, tag):
    """统一的配置获取方法"""
    if hasattr(self.application, 'start_state_manager'):
        return self.application.start_state_manager.get_instance_config(tag)
    return None

def _get_parallelism(self, tag):
    """统一的并行度获取方法"""
    if hasattr(self.application, 'start_state_manager'):
        return self.application.start_state_manager.get_parallelism(tag)
    return ModelParallelism(1, 1)
```

### 3. 提供默认配置

为没有配置管理器的情况提供默认配置：

```python
def _get_default_config(self):
    class InstanceConfig:
        def __init__(self):
            self.instance_type = "Splitwise"
            self.max_batch_size = 64
            self.max_batch_tokens = 4096
            self.max_preemptions = 3
    
    return InstanceConfig()
```

## 调试

### 查看配置

```python
# 打印配置信息
manager = app.start_state_manager
prompt_cfg = manager.get_instance_config("prompt")

print(f"Instance type: {prompt_cfg.instance_type}")
print(f"Max batch size: {prompt_cfg.max_batch_size}")
print(f"Max batch tokens: {prompt_cfg.max_batch_tokens}")

parallelism = manager.get_parallelism("prompt")
print(f"Pipeline parallelism: {parallelism.pipeline_parallelism}")
print(f"Tensor parallelism: {parallelism.tensor_parallelism}")
```

### 验证配置

```python
# 验证配置是否正确加载
assert hasattr(app, 'start_state_manager')
assert app.start_state_manager is not None

prompt_cfg = app.start_state_manager.get_instance_config("prompt")
assert prompt_cfg is not None
assert prompt_cfg.max_batch_size > 0
```

## 总结

配置管理系统提供了：

✅ **自动配置获取** - 从启动配置自动获取实例配置和并行度  
✅ **集中管理** - 所有配置在一个地方定义  
✅ **一致性保证** - 新实例与初始实例配置一致  
✅ **简化代码** - 调度器无需硬编码配置  
✅ **灵活性** - 支持多种配置类型  
✅ **可靠性** - 提供默认配置回退机制  

这使得扩缩容管理更加健壮和易于维护。

