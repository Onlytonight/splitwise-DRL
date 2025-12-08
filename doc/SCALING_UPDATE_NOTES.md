# 扩缩容管理器更新说明

## 版本 2.0 - 2024

### 主要更新

#### 1. 一个服务器一个实例

**原则：** 每个服务器只运行一个实例，简化资源管理。

**影响：**
- 扩容时自动创建新服务器
- 缩容时自动移除空闲服务器
- 资源隔离更好，管理更简单

#### 2. 全流程方法

新增两个高级方法，简化扩缩容操作：

**`scale_up_full()`** - 一行代码完成扩容
- 自动创建服务器
- 在服务器上创建实例
- 使用服务器的所有处理器

**`scale_down_full()`** - 一行代码完成缩容
- 排空并移除实例
- 自动移除空闲服务器
- 完全自动化

### API 变化

#### 新增 API

```python
# 全流程扩容（推荐）
server, instance = scaling_manager.scale_up_full(
    instance_cfg=config,
    parallelism=parallelism,
    tag="prompt",
    server_sku="A100"  # 可选
)

# 全流程缩容（推荐）
scaling_manager.scale_down_full(instance)
```

#### 现有 API（仍然可用）

原有的分步方法仍然可用，但推荐使用全流程方法：

```python
# 分步扩容（高级用法）
server = scaling_manager.scale_up_server(server_cfg)
instance = scaling_manager.scale_up_instance(
    instance_cfg=config,
    processors=server.processors,
    parallelism=parallelism,
    tag="prompt"
)

# 分步缩容（高级用法）
scaling_manager.scale_down_instance(instance)
if len(server.instances) == 0:
    scaling_manager.scale_down_server(server)
```

### 内部实现变化

#### 1. 自动服务器管理

- 扩容时自动创建服务器
- 缩容时自动检查并移除空闲服务器
- 通过 `_pending_server_scale_downs` 跟踪待缩容的服务器

#### 2. 流程简化

**扩容流程：**
```
scale_up_full() 
  → scale_up_server() 
  → scale_up_instance()
  → 延迟激活
```

**缩容流程：**
```
scale_down_full() 
  → scale_down_instance() 
  → 等待排空
  → _complete_scale_down()
  → 自动 scale_down_server()
```

### 迁移指南

#### 从 v1.0 迁移

**之前的代码：**

```python
# 需要手动管理服务器和实例
server = find_or_create_server()
instance = scaling_manager.scale_up_instance(
    instance_cfg=config,
    processors=server.processors,
    parallelism=parallelism,
    tag="prompt"
)

# 缩容也需要手动处理
scaling_manager.scale_down_instance(instance)
# 手动检查并移除服务器
if should_remove_server:
    scaling_manager.scale_down_server(server)
```

**现在的代码：**

```python
# 一行代码搞定
server, instance = scaling_manager.scale_up_full(
    instance_cfg=config,
    parallelism=parallelism,
    tag="prompt"
)

# 缩容也是一行
scaling_manager.scale_down_full(instance)
```

#### 配置文件

**无需修改**，配置文件保持不变：

```yaml
applications:
  - scaling_manager:
      scale_up_delay: 10.0
      drain_check_interval: 1.0
```

### 优势

#### 1. 简化使用

- **之前**：需要 5-10 行代码完成扩缩容
- **现在**：只需 1 行代码

#### 2. 减少错误

- 自动管理服务器，避免遗漏
- 自动检查服务器状态
- 自动清理资源

#### 3. 一致性保证

- 确保一个服务器一个实例
- 统一的扩缩容流程
- 更容易理解和维护

#### 4. 更好的资源管理

- 服务器与实例生命周期绑定
- 自动释放未使用的服务器
- 避免资源浪费

### 示例对比

#### 扩容对比

**v1.0 - 复杂：**

```python
def scale_up(self):
    # 1. 查找或创建服务器
    server = self._find_available_server()
    if server is None:
        server_cfg = self.cluster.get_server_config("A100")
        server = self.scaling_manager.scale_up_server(server_cfg)
    
    # 2. 创建实例
    instance = self.scaling_manager.scale_up_instance(
        instance_cfg=self._get_config(),
        processors=server.processors,
        parallelism=ModelParallelism(1, 1),
        tag="prompt"
    )
    
    return instance
```

**v2.0 - 简单：**

```python
def scale_up(self):
    # 一行搞定
    server, instance = self.scaling_manager.scale_up_full(
        instance_cfg=self._get_config(),
        parallelism=ModelParallelism(1, 1),
        tag="prompt"
    )
    return instance
```

#### 缩容对比

**v1.0 - 需要手动管理：**

```python
def scale_down(self, instance):
    # 1. 缩容实例
    servers = list(instance.servers)
    self.scaling_manager.scale_down_instance(instance)
    
    # 2. 手动等待并检查
    # （需要额外的回调机制）
    
    # 3. 手动移除服务器
    for server in servers:
        if len(server.instances) == 0:
            self.scaling_manager.scale_down_server(server)
```

**v2.0 - 全自动：**

```python
def scale_down(self, instance):
    # 一行搞定，自动处理所有流程
    self.scaling_manager.scale_down_full(instance)
```

### 兼容性

#### 向后兼容

- ✅ 所有 v1.0 的方法仍然可用
- ✅ 现有代码无需修改即可运行
- ✅ 配置文件格式不变

#### 推荐升级

虽然向后兼容，但**强烈推荐**使用新的全流程方法：

1. **更简单**：代码更少，更易读
2. **更可靠**：自动管理，减少错误
3. **更一致**：统一的扩缩容模式

### 测试建议

#### 测试场景

1. **基本扩缩容**
   - 测试 `scale_up_full()` 正确创建服务器和实例
   - 测试 `scale_down_full()` 正确排空并移除

2. **并发扩缩容**
   - 同时扩容多个实例
   - 同时缩容多个实例

3. **边界情况**
   - 配置缺失
   - 服务器创建失败
   - 实例无法排空

4. **资源清理**
   - 确保服务器正确移除
   - 确保没有资源泄漏

### 已知限制

1. **一个服务器一个实例**
   - 如果需要多个实例共享服务器，需要使用分步方法
   - 未来版本可能支持配置

2. **服务器 SKU 选择**
   - 目前使用配置中的第一个或指定的 SKU
   - 未来可能支持智能选择

### 文档更新

所有文档已更新以反映新的 API：

- ✅ `SCALING_QUICKSTART.md` - 快速入门
- ✅ `SCALING_MANAGER_GUIDE.md` - 详细指南
- ✅ `SCALING_USAGE_SUMMARY.md` - 使用总结
- ✅ `example_autoscaling_scheduler.py` - 代码示例

### 反馈

如有问题或建议，请查看文档或提交反馈。

---

**推荐行动：**

1. 阅读 `SCALING_QUICKSTART.md` 快速了解新功能
2. 查看 `example_autoscaling_scheduler.py` 了解实际用法
3. 在新代码中使用 `scale_up_full()` 和 `scale_down_full()`
4. 逐步迁移现有代码（可选，但推荐）

