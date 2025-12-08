# 扩缩容管理器使用总结

## 核心原则

**一个服务器一个实例**

## 推荐用法

### 扩容

```python
# 全流程扩容：自动创建服务器 + 实例
from model import ModelParallelism

server, instance = scaling_manager.scale_up_full(
    instance_cfg=instance_cfg,
    parallelism=ModelParallelism(pipeline_parallelism=1, tensor_parallelism=1),
    tag="prompt",  # 或 "token"
    server_sku="A100"  # 可选
)
```

**流程：**
1. 自动创建新服务器
2. 在服务器上创建实例（使用所有处理器）
3. 实例延迟后变为 ACTIVE

### 缩容

```python
# 全流程缩容：自动排空实例 + 移除服务器
scaling_manager.scale_down_full(instance)
```

**流程：**
1. 实例停止接收新任务
2. 等待现有任务完成
3. 移除实例
4. 自动移除服务器（如果没有其他实例）

## 配置

```yaml
applications:
  - application_id: 0
    scaling_manager:
      scale_up_delay: 10.0           # 扩容延迟（秒）
      drain_check_interval: 1.0      # 排空检查间隔（秒）
```

## 在调度器中使用

```python
class MyScheduler(Scheduler):
    def schedule(self, request):
        # 定期检查负载
        if self.request_count % 100 == 0:
            self._check_and_scale()
        
        # 调度时自动过滤扩缩容中的实例
        schedulable = self.get_schedulable_instances()
        # ... 正常调度逻辑 ...
    
    def _check_and_scale(self):
        # 扩容 - 自动从配置获取
        if self._is_overloaded():
            instance_cfg = self._get_instance_config("prompt")
            parallelism = self._get_parallelism("prompt")
            
            server, instance = self.application.scaling_manager.scale_up_full(
                instance_cfg=instance_cfg,
                parallelism=parallelism,
                tag="prompt"
            )
        
        # 缩容
        elif self._is_underloaded() and len(self.instances) > 1:
            least_loaded = min(self.instances, key=lambda i: len(i.pending_queue))
            self.application.scaling_manager.scale_down_full(least_loaded)
    
    def _get_instance_config(self, tag):
        """从启动状态配置获取实例配置"""
        if hasattr(self.application, 'start_state_manager'):
            return self.application.start_state_manager.get_instance_config(tag)
        return default_config
    
    def _get_parallelism(self, tag):
        """从启动状态配置获取并行度"""
        if hasattr(self.application, 'start_state_manager'):
            return self.application.start_state_manager.get_parallelism(tag)
        return ModelParallelism(1, 1)
```

## 关键特性

✅ **一行代码完成扩缩容**  
✅ **自动状态管理**  
✅ **调度器自动过滤**  
✅ **优雅排空任务**  
✅ **配置驱动**  

## 状态

| 状态 | 说明 | 可调度 |
|------|------|--------|
| ACTIVE | 正常运行 | ✅ |
| SCALING_UP | 正在启动 | ❌ |
| DRAINING | 排空中 | ❌ |
| SCALING_DOWN | 已缩容 | ❌ |

## 详细文档

- **快速入门**：`SCALING_QUICKSTART.md`
- **详细指南**：`SCALING_MANAGER_GUIDE.md`
- **实现总结**：`SCALING_IMPLEMENTATION_SUMMARY.md`
- **代码示例**：`example_autoscaling_scheduler.py`

