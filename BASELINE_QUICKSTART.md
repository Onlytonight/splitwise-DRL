# 基线策略快速入门

本文档提供快速入门指南，帮助你在5分钟内运行基线扩缩容策略。

## 快速开始

### 1. 运行 HeteroScale（推荐）

```bash
python run.py +experiment=baseline_heteroscale
```

这将运行生产级的 HeteroScale 策略，结合了比例控制和延迟触发机制。

### 2. 运行基于显存利用率的策略

```bash
python run.py +experiment=baseline_utilization
```

### 3. 运行基于队列长度的策略

```bash
python run.py +experiment=baseline_queue
```

## 查看结果

运行完成后，结果保存在 `outputs/` 目录下：

```bash
# 查看汇总结果
cat outputs/baseline_heteroscale/YYYY-MM-DD/HH-MM-SS/summary.csv

# 查看奖励曲线
cat outputs/baseline_heteroscale/YYYY-MM-DD/HH-MM-SS/reward_heteroscale.csv
```

## 自定义配置

### 调整决策间隔

```bash
python run.py +experiment=baseline_heteroscale \
  simulator.decision_interval=5  # 每5秒做一次决策
```

### 调整实例数限制

```bash
python run.py +experiment=baseline_heteroscale \
  simulator.max_total_instances=100 \
  simulator.min_instances_per_pool=2
```

### 调整 HeteroScale 参数

```bash
python run.py +experiment=baseline_heteroscale \
  simulator.policy_config.target_tps_per_instance=150 \
  simulator.policy_config.pd_ratio=0.5 \
  simulator.policy_config.tbt_slo=60.0
```

### 使用自定义 trace

```bash
python run.py +experiment=baseline_heteroscale \
  trace.filename=my_trace_name
```

## 批量运行所有基线

```bash
# Linux/Mac
chmod +x scripts/run_baselines.sh
./scripts/run_baselines.sh

# Windows
bash scripts/run_baselines.sh
```

## 关键参数说明

### HeteroScale 策略

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `target_tps_per_instance` | 目标单实例TPS | 100 | 80-150 |
| `pd_ratio` | P/D比例 (1:X) | 0.33 (1:3) | 0.25-0.5 |
| `scale_out_threshold` | 扩容阈值 | 0.1 | 0.05-0.2 |
| `scale_in_threshold` | 缩容阈值 | 0.1 | 0.05-0.2 |
| `tbt_slo` | TBT SLO (ms) | 50.0 | 30-100 |
| `emergency_scale_ratio` | 紧急扩容幅度 | 0.2 (20%) | 0.1-0.3 |

### UtilizationBased 策略

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `upper_threshold` | 显存利用率上限 | 0.8 | 0.7-0.9 |
| `lower_threshold` | 显存利用率下限 | 0.3 | 0.2-0.4 |
| `scale_step` | 扩缩容步长 | 1 | 1-5 |

### QueueBased 策略

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `prompt_queue_upper` | Prompt队列上限 | 1000 | 500-2000 |
| `prompt_queue_lower` | Prompt队列下限 | 100 | 50-200 |
| `token_queue_upper` | Token队列上限 | 5000 | 2000-10000 |
| `token_queue_lower` | Token队列下限 | 500 | 200-1000 |

## 对比不同策略

```bash
# 运行三个基线策略
python run.py +experiment=baseline_heteroscale
python run.py +experiment=baseline_utilization
python run.py +experiment=baseline_queue

# 对比结果（手动比较 summary.csv）
```

关键对比指标：
- **TTFT P99**: 首token延迟（`ttft` 是列表 `[p50, p90, p99]`，使用 P99 值）
- **TBT P99**: 字间延迟（`tbt` 是列表 `[p50, p90, p99]`，使用 P99 值）
- **平均实例数**: 资源成本
- **队列长度**: 系统负载

## 与 RL 方法对比

```bash
# 运行 SAC（RL方法）
python run.py +experiment=your_sac_experiment

# 运行 HeteroScale（基线）
python run.py +experiment=baseline_heteroscale

# 对比结果
```

## 常见问题

**Q: 基线策略需要训练多个 epoch 吗？**

A: 不需要。基线策略是规则策略，运行一次即可。配置文件中已设置 `trace_epochs: 1`。

**Q: 如何加速测试？**

A: 使用较短的 trace 和较小的实例数限制：

```bash
python run.py +experiment=baseline_heteroscale \
  trace.filename=short_trace \
  simulator.max_total_instances=50 \
  simulator.decision_interval=5
```

**Q: 输出文件在哪里？**

A: 默认在 `outputs/baseline_xxx/` 目录下，按时间戳组织。

**Q: 如何调整日志级别？**

A: 在配置文件中设置或通过环境变量：

```bash
# 详细日志
python run.py +experiment=baseline_heteroscale \
  hydra.verbose=true

# 静默模式
python run.py +experiment=baseline_heteroscale \
  hydra.verbose=false
```

## 下一步

- 📖 阅读详细文档：`doc/BASELINE_POLICIES.md`
- 🔧 添加自定义策略：参考文档第 4 节
- 📊 分析结果：使用 `notebooks/` 下的可视化脚本
- 🚀 调优参数：根据你的工作负载调整策略参数

## 需要帮助？

- 检查 `simulator.log` 查看详细日志
- 查看 `reward_xxx.csv` 了解策略表现
- 阅读完整文档：`doc/BASELINE_POLICIES.md`
