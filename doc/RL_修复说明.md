# 强化学习状态维度和归一化修复说明

## 修复的问题

### 1. 状态维度不一致问题 ✅
**问题描述：** 
- 代码中 `feature_dim` 设置为 16，但实际特征有 20 维
- Performance 部分包含 6 个指标（3个TTFT + 3个TBT），而不是注释中说的 2 个
- 导致每次状态更新时维度不匹配，引发错误

**修复内容：**
- `RL/state.py` 第 20 行：`feature_dim = 16` → `feature_dim = 20`
- `simulator.py` 第 284 行：`state_dim = 68` → `state_dim = 80`（4层堆叠 × 20维特征）

**维度构成：**
```
特征索引 [0-3]:   Workload (负载)      - 4维
特征索引 [4-6]:   Queue (队列)         - 3维
特征索引 [7-9]:   Instance Counts (实例数) - 3维
特征索引 [10-12]: Utilization (利用率)  - 3维
特征索引 [13]:    Network Util (网络)   - 1维
特征索引 [14-19]: Performance (性能)    - 6维 (3个TTFT + 3个TBT)
-------------------------------------------
总计: 20维
堆叠4层后: 80维
```

### 2. 状态值归一化问题 ✅
**问题描述：**
- 原归一化函数只处理前 10 个特征，后 10 个特征（索引 10-19）直接复制
- 导致部分特征的数值范围差异很大，影响 PPO 训练效果

**修复内容：**
改进了 `_normalize()` 函数，对所有 20 维特征进行适当归一化：

#### 详细归一化策略：

**A. 负载特征 [0-3]:**
```python
norm_vec[0] = np.log1p(raw_vector[0]) / 10.0  # RPS: log缩放
norm_vec[1] = np.log1p(raw_vector[1]) / 10.0  # Token Rate: log缩放
norm_vec[2] = np.clip(raw_vector[2] / 4096.0, 0, 1)  # Prompt Length: 除以最大值
norm_vec[3] = np.clip(raw_vector[3] / 2048.0, 0, 1)  # Output Length: 除以最大值
```
- **原理：** RPS 和 Token Rate 是长尾分布，使用 log1p 压缩大数值
- **效果：** 将 0~10000 的范围压缩到 0~1 之间

**B. 队列特征 [4-6]:**
```python
norm_vec[4] = np.log1p(raw_vector[4]) / 10.0  # Prompt Queue: log缩放
norm_vec[5] = np.log1p(raw_vector[5]) / 10.0  # Decode Queue: log缩放
norm_vec[6] = np.tanh(raw_vector[6] / 10.0)   # Wait Time: tanh压缩
```
- **原理：** 队列长度也是长尾分布，等待时间用 tanh 压缩到 [-1, 1]
- **效果：** 避免队列爆炸时数值过大

**C. 资源特征 [7-12]:**
```python
norm_vec[7-9] = np.clip(raw_vector[7-9] / MAX_INSTANCES, 0, 1)  # 实例数归一化
norm_vec[10-12] = np.clip(raw_vector[10-12], 0, 1)  # 利用率 (已在0-1之间)
```
- **原理：** 实例数除以最大集群规模，利用率本身就在 0-1 之间
- **效果：** 统一到 [0, 1] 范围

**D. 网络特征 [13]:**
```python
norm_vec[13] = np.clip(raw_vector[13], 0, 1)  # 网络利用率
```

**E. 性能特征 [14-19]:**
```python
for i in range(14, 20):
    norm_vec[i] = np.tanh(raw_vector[i])  # TTFT和TBT比率
```
- **原理：** 这些是 actual/SLO 的比率，< 1 表示满足 SLO，> 1 表示违反
- **效果：** tanh 将 0-2 的范围映射到约 0-1 之间，保持单调性

## 修改的文件清单

1. **RL/state.py**
   - 修正 `feature_dim` 从 16 → 20
   - 重写 `_normalize()` 函数，添加完整的归一化逻辑
   - 添加详细注释说明各特征的含义和索引

2. **simulator.py**
   - 修正 `state_dim` 从 68 → 80
   - 更新注释说明

## 预期效果

### 1. 修复维度错误
- ✅ 不再出现维度不匹配的运行时错误
- ✅ PPO 网络输入维度正确（80维）

### 2. 改善训练稳定性
- ✅ 所有特征值归一化到相似范围（大部分在 0-1 之间）
- ✅ 避免某些特征主导梯度更新
- ✅ 提高 PPO 训练收敛速度

### 3. 提升策略质量
- ✅ 网络能够更好地学习各特征之间的关系
- ✅ 减少数值不稳定导致的策略震荡

## 测试建议

运行以下测试验证修复：

```python
from RL.state import RLStateCollector

# 1. 检查维度
collector = RLStateCollector(cluster, router, applications, stack_size=4)
print(f"Feature dim: {collector.feature_dim}")  # 应输出 20

# 2. 检查堆叠后的状态维度
state = collector.get_stacked_state(current_time, interval)
print(f"Stacked state dim: {state.shape}")  # 应输出 (80,)

# 3. 检查归一化值范围
print(f"State min: {state.min():.4f}, max: {state.max():.4f}")  
# 应该在合理范围内，大部分特征在 [-1, 1] 或 [0, 1]
```

## 注意事项

⚠️ **重要：** 如果已经训练了 PPO 模型，需要重新训练，因为：
1. 输入维度从 68 改为 80，旧模型不兼容
2. 归一化策略改变，旧经验的数值范围不同

## 后续优化建议

1. **自适应归一化：** 可以考虑使用 Running Mean/Std 进行动态归一化
2. **特征工程：** 可能需要根据实际数据分布调整归一化参数（如 log1p 的除数）
3. **监控指标：** 建议记录训练时各特征的实际分布，确保归一化效果

