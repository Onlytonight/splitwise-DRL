# 自动扩缩容策略对比图表生成指南

本指南说明如何使用提供的脚本生成不同自动扩缩容策略的对比图表。

---

## 目录结构

根据当前的配置文件 `configs/config.yaml`，实验结果的目录结构为：

```
results/
└── {seed}/                          # 随机种子（如 0）
    └── {start_state}/               # 启动状态（如 splitwise_25_15）
        └── {trace}/                 # 流量追踪（如 rr_conv_40）
            └── {cluster}/           # 集群配置（如 0_40）
                └── {model}/         # 模型名称（如 bloom-176b）
                    └── {scheduler}/ # 调度器（如 mixed_pool）
                        └── {policy}/ # 扩缩容策略（如 heteroscale）
                            ├── summary.csv
                            ├── request.csv
                            ├── scaling_app_0.log
                            └── ...
```

**示例路径**：
```
results/0/splitwise_25_15/rr_conv_40/0_40/bloom-176b/mixed_pool/heteroscale/summary.csv
```

---

## 可用的绘图脚本

### 1. `compare_autoscaling_policies.py` - 完整对比脚本

**功能**：
- 生成多种对比图表（折线图、网格图、柱状图）
- 支持多个分位数（P50, P90, P99）
- 自动计算归一化指标（slowdown）
- 生成数据摘要 CSV

**使用方法**：

1. **修改配置参数**（在脚本的 `main()` 函数中）：

```python
# 扩缩容策略列表
policies = [
    "heteroscale",      # HeteroScale (TPS + Latency)
    "hpa_gpu",          # HPA-GPU (GPU Utilization)
    "independent_tps",  # Independent TPS
    "pure_latency",     # Pure Latency
]

# 流量追踪列表
trace_type = "conv"  # 或 "code"
trace_rates = range(10, 110, 10)  # 10, 20, 30, ..., 100
traces = [f"rr_{trace_type}_{rate}" for rate in trace_rates]

# 其他配置
seed = 0
start_state = "splitwise_25_15"
cluster = "0_40"
scheduler = "mixed_pool"
model = "bloom-176b"
```

2. **运行脚本**：

```bash
cd notebooks
python compare_autoscaling_policies.py
```

3. **输出**：

图表保存在 `plots/autoscaling_comparison/` 目录：
- `ttft_slowdown_comparison.png`: TTFT Slowdown 对比
- `tbt_slowdown_comparison.png`: TBT Slowdown 对比
- `e2e_slowdown_comparison.png`: E2E Slowdown 对比
- `metrics_grid_p90.png`: 多指标网格对比（P90）
- `bar_comparison_tbt_p90.png`: 柱状图对比
- `results_summary.csv`: 数据摘要

---

### 2. `quick_compare.py` - 快速对比脚本（推荐）

**功能**：
- 简化版，专注于核心对比
- 直接使用 summary.csv（无需性能模型）
- 快速生成折线图和汇总表

**使用方法**：

1. **修改配置区域**（在脚本开头）：

```python
# ==================== 配置区域 ====================

# 1. 基础路径
RESULTS_DIR = "../results"
PLOTS_DIR = "../plots/autoscaling_quick/"

# 2. 实验配置
SEED = 0
START_STATE = "splitwise_25_15"
CLUSTER = "0_40"
SCHEDULER = "mixed_pool"
MODEL = "bloom-176b"

# 3. 扩缩容策略
POLICIES = [
    "heteroscale",
    "hpa_gpu",
    "independent_tps",
    "pure_latency",
]

# 4. 流量追踪
TRACE_TYPE = "conv"  # 或 "code"
TRACE_RATES = [10, 20, 30, 40, 50]  # 请求速率列表
```

2. **运行脚本**：

```bash
cd notebooks
python quick_compare.py
```

3. **输出**：

```
正在收集数据...
✓ heteroscale - rr_conv_10
✓ heteroscale - rr_conv_20
✓ hpa_gpu - rr_conv_10
...

✓ 成功收集 20 条记录

生成对比图表...
✓ 保存: ttft_comparison.png
✓ 保存: tbt_comparison.png
✓ 保存: e2e_comparison.png
✓ 保存汇总表: summary_table.csv

============================================================
汇总表 (平均值)
============================================================
          Policy  TTFT P90  TBT P90  E2E P90
     heteroscale    0.1234   0.0456   1.2345
         hpa_gpu    0.1456   0.0567   1.4567
  independent_tps    0.1567   0.0678   1.5678
    pure_latency    0.1678   0.0789   1.6789
============================================================
```

图表保存在 `plots/autoscaling_quick/` 目录。

---

## 生成的图表类型

### 1. 折线图对比（按分位数）

- **横轴**：请求速率（Request Rate）
- **纵轴**：性能指标（TTFT, TBT, E2E）
- **不同颜色的线**：不同的扩缩容策略
- **红色虚线**：SLO 阈值

**示例**：
```
TTFT Slowdown (P50) | TTFT Slowdown (P90) | TTFT Slowdown (P99)
--------------------+--------------------+--------------------
       📈           |        📈          |        📈
    各策略对比       |     各策略对比      |     各策略对比
       ---          |        ---         |        ---
```

### 2. 多指标网格对比

- 在同一行显示多个指标（TTFT, TBT, E2E）
- 固定分位数（如 P90）
- 便于横向对比不同指标

### 3. 柱状图对比

- 聚合所有流量的平均值
- 直观对比各策略的整体表现
- 柱子上方标注具体数值

### 4. 汇总表格（CSV）

包含所有策略在不同流量下的详细数据，可用于后续分析。

---

## 常见配置示例

### 示例 1：对比所有基线策略

```python
POLICIES = [
    "heteroscale",
    "hpa_gpu",
    "independent_tps",
    "pure_latency",
    "periodic",
    "no_autoscaling"
]
```

### 示例 2：只对比核心策略

```python
POLICIES = [
    "heteroscale",
    "hpa_gpu",
    "independent_tps"
]
```

### 示例 3：测试不同流量类型

```python
# 对话流量
TRACE_TYPE = "conv"
TRACE_RATES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# 或代码生成流量
TRACE_TYPE = "code"
TRACE_RATES = [5, 10, 15, 20, 25, 30]
```

### 示例 4：不同集群配置

```python
# 小集群
CLUSTER = "0_20"

# 中等集群
CLUSTER = "0_40"

# 大集群
CLUSTER = "0_80"
```

---

## 故障排查

### 问题 1：找不到数据文件

**错误信息**：
```
警告: 路径不存在: results/0/splitwise_25_15/rr_conv_40/0_40/bloom-176b/mixed_pool/heteroscale
```

**解决方案**：
1. 检查实验是否已运行完成
2. 确认配置参数与实际路径一致
3. 检查路径拼写是否正确

**验证命令**：
```bash
# Windows
dir results\0\splitwise_25_15\rr_conv_40\0_40\bloom-176b\mixed_pool\

# Linux/Mac
ls results/0/splitwise_25_15/rr_conv_40/0_40/bloom-176b/mixed_pool/
```

### 问题 2：缺少 perf_model.csv

**错误信息**：
```
FileNotFoundError: data/perf_model.csv
```

**解决方案**：
- 使用 `quick_compare.py`（不需要 perf_model.csv）
- 或者确保 `data/perf_model.csv` 存在

### 问题 3：图表显示不全

**解决方案**：
- 调整 `figsize` 参数增大图表尺寸
- 减少要对比的策略数量
- 调整字体大小

---

## 自定义图表

### 修改图表尺寸

在脚本中找到 `plt.subplots` 并修改 `figsize`：

```python
# 原始
fig, axs = plt.subplots(1, 3, figsize=(14, 4))

# 增大
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
```

### 修改颜色方案

```python
# 在脚本开头添加
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')  # 或其他样式

# 或自定义颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
```

### 添加自定义指标

在 `plot_comparison` 函数中添加新的指标：

```python
# 示例：添加吞吐量对比
plot_comparison(df, 'throughput', 'Throughput (req/s)', 
               'Throughput Comparison', 'throughput_comparison.png')
```

---

## 批量生成图表

创建一个批处理脚本 `generate_all_plots.sh`（Linux/Mac）：

```bash
#!/bin/bash

echo "生成所有对比图表..."

# 对话流量
python quick_compare.py --trace-type conv --rates 10,20,30,40,50

# 代码生成流量
python quick_compare.py --trace-type code --rates 5,10,15,20,25

# 不同集群
python quick_compare.py --cluster 0_20
python quick_compare.py --cluster 0_40
python quick_compare.py --cluster 0_80

echo "完成！"
```

或 Windows 批处理 `generate_all_plots.bat`：

```batch
@echo off
echo 生成所有对比图表...

REM 对话流量
python quick_compare.py

REM 修改配置后再次运行...

echo 完成！
pause
```

---

## 高级用法

### 1. 生成动画（流量变化）

```python
import matplotlib.animation as animation

def animate_trace(results_df, policies):
    fig, ax = plt.subplots()
    
    def update(frame):
        ax.clear()
        trace = traces[frame]
        data = results_df[results_df['trace'] == trace]
        # ... 绘图代码 ...
    
    anim = animation.FuncAnimation(fig, update, frames=len(traces))
    anim.save('policy_comparison.gif', writer='pillow')
```

### 2. 生成交互式图表（Plotly）

```python
import plotly.express as px

fig = px.line(results_df, x='rate', y='tbt_times_p90', color='policy',
              title='TBT P90 Comparison')
fig.write_html('interactive_comparison.html')
```

### 3. 统计显著性测试

```python
from scipy import stats

# 比较两个策略
policy1_data = results_df[results_df['policy'] == 'heteroscale']['tbt_times_p90']
policy2_data = results_df[results_df['policy'] == 'hpa_gpu']['tbt_times_p90']

t_stat, p_value = stats.ttest_ind(policy1_data, policy2_data)
print(f"T-statistic: {t_stat}, P-value: {p_value}")
```

---

## 输出示例

### 图表示例

**折线图**：
```
         TBT P90 (seconds)
    2.0 |                    
        |      ●---●         ← pure_latency (波动大)
    1.5 |   ●      ●--●     
        | ●--●--●         ● ← independent_tps
    1.0 |●--●--●--●--●--●   ← heteroscale (稳定)
        |●--●--●--●--●--●   ← hpa_gpu
    0.5 |___________________
        10  20  30  40  50  (req/s)
```

**汇总表**：
```
Policy             TTFT P90   TBT P90   E2E P90   违规率
===============================================================
heteroscale        0.1234     0.0456    1.2345    2.3%  ✓
hpa_gpu            0.1456     0.0567    1.4567    8.7%
independent_tps    0.1567     0.0678    1.5678    15.2%
pure_latency       0.1678     0.0789    1.6789    23.4%
```

---

## 参考资料

- **原始脚本**: `notebooks/generate_plots.py`
- **工具函数**: `notebooks/utils.py`
- **性能模型**: `notebooks/perf_model.py`
- **基线策略文档**: `doc/BASELINE_POLICIES.md`

---

## 联系与反馈

如有问题或建议，请参考项目文档或联系维护者。

