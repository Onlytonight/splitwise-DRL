"""
自动扩缩容 vs 非自动扩缩容对比分析

对比两个方面：
1. 总机器数随时间变化（autoscaling vs 固定8P+4T）
2. 性能指标对比（autoscaling vs 固定3P+3T）
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 导入性能模型
sys.path.append('.')
from perf_model import PerfModel

# ============================================================================
# 配置
# ============================================================================
results_dir = "../results/motivation/0/day_30"
perf_model_path = "../data/perf_model.csv"
plots_dir = "../plots/autoscaling_comparison"
os.makedirs(plots_dir, exist_ok=True)

# 可选：只分析前N秒的数据（设为 None 表示分析全部数据）
MAX_TIME = 600  # 例如: 600 表示只看前10分钟

# 数据路径
paths = {
    'autoscaling': f"{results_dir}/5_5/bloom-176b/mixed_pool/heteroscale",
    'fixed_8_4': f"{results_dir}/10_10/bloom-176b/mixed_pool/no_autoscaling",
    'fixed_3_3': f"{results_dir}/2_2/bloom-176b/mixed_pool/no_autoscaling",
}

# 模型配置
model = "bloom-176b"
normalize_hardware = "h100-80gb"
normalize_tp = 8

# 可视化配置
METRICS = {
    'e2e': {'column': 'response_times', 'label': 'E2E', 'slowdown': 'e2e_slowdown'},
    'ttft': {'column': 'ttft_times', 'label': 'TTFT', 'slowdown': 'ttft_slowdown'},
    'tbt': {'column': 'tbt_times', 'label': 'TBT', 'slowdown': 'tbt_slowdown'},
}
QUANTILES = [0.5, 0.9, 0.99]
QUANTILE_LABELS = ['p50', 'p90', 'p99']

print("=" * 70)
print("自动扩缩容 vs 非自动扩缩容对比分析")
print("=" * 70)

# ============================================================================
# 图1: 总机器数随时间变化
# ============================================================================
print("\n[1/2] 加载扩缩容数据...")

def load_scaling_data(path, max_time=None):
    """
    加载 scaling 日志数据
    
    Args:
        path: 结果目录路径
        max_time: 可选，只加载 time <= max_time 的数据
    
    Returns:
        DataFrame: 时间序列的实例数状态
    """
    scaling_file = f"{path}/scaling/0.csv"
    try:
        df = pd.read_csv(scaling_file)
        
        # 可选：只取前N秒的数据
        if max_time is not None:
            df = df[df['time'] <= max_time]
        
        # 日志格式：time,total_instances,prompt_instances,token_instances,pending_queue_length,prefill_tps,decode_tps,total_tps,reason
        # 一次循环记录两次：before 和 after，我们取 after 的记录（或者去重）
        # 为了绘图简洁，可以按时间分组，取最后一条（after）
        if 'reason' in df.columns:
            # 按时间分组，保留每个时间点的最后一条记录（after）
            df = df.groupby('time', as_index=False).last()
        
        return df
    except Exception as e:
        print(f"  ✗ 无法加载 {scaling_file}: {e}")
        return None

# 加载自动扩缩容数据
autoscaling_df = load_scaling_data(paths['autoscaling'], max_time=MAX_TIME)
if autoscaling_df is not None:
    print(f"  ✓ 自动扩缩容: {len(autoscaling_df)} 条记录 (时间点: {autoscaling_df['time'].min():.1f}s - {autoscaling_df['time'].max():.1f}s)")
    print(f"    最大实例数: {autoscaling_df['total_instances'].max()}, 最小实例数: {autoscaling_df['total_instances'].min()}")
    if 'pending_queue_length' in autoscaling_df.columns:
        print(f"    最大队列长度: {autoscaling_df['pending_queue_length'].max()}, 平均队列长度: {autoscaling_df['pending_queue_length'].mean():.1f}")
    if 'total_tps' in autoscaling_df.columns:
        print(f"    最大 TPS: {autoscaling_df['total_tps'].max():.1f}, 平均 TPS: {autoscaling_df['total_tps'].mean():.1f}")
        print(f"    平均 Prefill TPS: {autoscaling_df['prefill_tps'].mean():.1f}, 平均 Decode TPS: {autoscaling_df['decode_tps'].mean():.1f}")
else:
    print("  ✗ 自动扩缩容数据加载失败")

# 对于固定配置，创建固定值数据
def create_fixed_scaling_data(path, prompt_count, token_count):
    """为固定配置创建扩缩容数据"""
    # 读取一个数据文件来获取时间范围
    detailed_file = f"{path}/detailed/0.csv"
    try:
        df = pd.read_csv(detailed_file)
        max_time = 600
    except:
        max_time = 600
    
    # 创建固定值数据
    total = prompt_count + token_count
    return pd.DataFrame({
        'time': [0, max_time],
        'total_instances': [total, total],
        'prompt_instances': [prompt_count, prompt_count],
        'token_instances': [token_count, token_count],
        'pending_queue_length': [0, 0],  # 固定配置没有队列长度数据
        'prefill_tps': [0.0, 0.0],  # 固定配置没有 TPS 数据
        'decode_tps': [0.0, 0.0],
        'total_tps': [0.0, 0.0],
        'reason': ['fixed', 'fixed'],
    })

fixed_8_4_df = create_fixed_scaling_data(paths['fixed_8_4'], 10, 10)
print(f"  ✓ 固定配置(10P+10T): {len(fixed_8_4_df)} 条记录")

# 绘制图1：总机器数随时间变化
print("\n绘制图1: 总机器数随时间变化...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Autoscaling vs Fixed Configuration: Instance Count Over Time', 
             fontsize=16, fontweight='bold')

# 子图1: 总实例数
ax1 = axes[0]
if autoscaling_df is not None:
    ax1.plot(autoscaling_df['time'], autoscaling_df['total_instances'], 
             label='Autoscaling (5P+5T → Dynamic)', linewidth=2, color='#2ca02c', marker='o', markersize=3)
ax1.plot(fixed_8_4_df['time'], fixed_8_4_df['total_instances'], 
         label='Fixed (10P+10T)', linewidth=2, color='#ff7f0e', linestyle='--')
ax1.set_xlabel('Time (seconds)', fontweight='bold')
ax1.set_ylabel('Total Instances', fontweight='bold')
ax1.set_title('Total Instance Count Comparison', fontsize=12)
ax1.legend(loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3)

# 子图2: Prompt 和 Token 实例数分开显示
ax2 = axes[1]
if autoscaling_df is not None:
    ax2.plot(autoscaling_df['time'], autoscaling_df['prompt_instances'], 
             label='Autoscaling - Prompt', linewidth=2, color='#1f77b4', marker='s', markersize=3)
    ax2.plot(autoscaling_df['time'], autoscaling_df['token_instances'], 
             label='Autoscaling - Token', linewidth=2, color='#d62728', marker='^', markersize=3)
ax2.plot(fixed_8_4_df['time'], fixed_8_4_df['prompt_instances'], 
         label='Fixed - Prompt (10)', linewidth=2, color='#1f77b4', linestyle='--', alpha=0.6)
ax2.plot(fixed_8_4_df['time'], fixed_8_4_df['token_instances'], 
         label='Fixed - Token (10)', linewidth=2, color='#d62728', linestyle='--', alpha=0.6)
ax2.set_xlabel('Time (seconds)', fontweight='bold')
ax2.set_ylabel('Instance Count', fontweight='bold')
ax2.set_title('Prompt & Token Instance Count Comparison', fontsize=12)
ax2.legend(loc='best', framealpha=0.9, ncol=2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path = f"{plots_dir}/instance_count_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"  ✓ 已保存: {save_path}")
plt.close()

# ============================================================================
# 图2: 性能指标对比 (p90 E2E)
# ============================================================================
print("\n[2/2] 加载性能数据...")

# 加载性能模型
perf_model = PerfModel(perf_model_path, init=True)
print("  ✓ 性能模型已加载")

def load_and_calculate_slowdown(path, name):
    """加载详细数据并计算 slowdown"""
    detailed_file = f"{path}/detailed/0.csv"
    try:
        df = pd.read_csv(detailed_file)
        print(f"  ✓ {name}: {len(df)} 条请求")
        
        # 计算 baseline 性能
        perf_model.add_baseline_perf(df, model, normalize_hardware, normalize_tp)
        df["baseline_e2e"] = df["baseline_ttft"] + df["baseline_tbt"] * (df["token_sizes"] - 1)
        
        # 计算 slowdown
        df["ttft_slowdown"] = df["ttft_times"] / df["baseline_ttft"]
        df["tbt_slowdown"] = df["tbt_times"] / df["baseline_tbt"]
        df["e2e_slowdown"] = df["response_times"] / df["baseline_e2e"]
        
        return df
    except Exception as e:
        print(f"  ✗ {name} 加载失败: {e}")
        return None

# 加载数据
autoscaling_perf = load_and_calculate_slowdown(paths['autoscaling'], "自动扩缩容")
fixed_3_3_perf = load_and_calculate_slowdown(paths['fixed_3_3'], "固定配置(2P+2T)")

# 计算分位数
def calculate_quantiles(df, metrics, quantiles):
    """计算所有指标的分位数"""
    results = {}
    for metric_name, metric_info in metrics.items():
        slowdown_col = metric_info['slowdown']
        results[metric_name] = {}
        for q in quantiles:
            q_label = f"p{int(q*100)}"
            results[metric_name][q_label] = df[slowdown_col].quantile(q)
    return results

if autoscaling_perf is not None:
    auto_quantiles = calculate_quantiles(autoscaling_perf, METRICS, QUANTILES)
    print(f"  ✓ 自动扩缩容分位数计算完成")
else:
    auto_quantiles = None

if fixed_3_3_perf is not None:
    fixed_quantiles = calculate_quantiles(fixed_3_3_perf, METRICS, QUANTILES)
    print(f"  ✓ 固定配置分位数计算完成")
else:
    fixed_quantiles = None

# 绘制图2：性能对比（可选指标）
print("\n绘制图2: 性能指标对比...")

def plot_performance_comparison(metric_key='e2e', quantile_key='p90'):
    """
    绘制性能对比图
    
    Args:
        metric_key: 'e2e', 'ttft', 或 'tbt'
        quantile_key: 'p50', 'p90', 或 'p99'
    """
    if auto_quantiles is None or fixed_quantiles is None:
        print("  ✗ 数据不完整，无法绘制")
        return
    
    metric_label = METRICS[metric_key]['label']
    
    # 准备数据
    configs = ['Autoscaling\n(5P+5T→Dynamic)', 'Fixed\n(2P+2T)']
    values = [
        auto_quantiles[metric_key][quantile_key],
        fixed_quantiles[metric_key][quantile_key]
    ]
    colors = ['#2ca02c', '#ff7f0e']
    
    # 创建条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(configs, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 设置标签和标题
    ax.set_ylabel(f'{metric_label} Slowdown', fontsize=13, fontweight='bold')
    ax.set_title(f'{quantile_key.upper()} {metric_label} Slowdown: Autoscaling vs Fixed Configuration',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # 设置y轴范围
    ax.set_ylim(bottom=0, top=max(values) * 1.2)
    
    plt.tight_layout()
    save_path = f"{plots_dir}/performance_{metric_key}_{quantile_key}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已保存: {save_path}")
    plt.close()

# 生成默认图（p90 E2E）
plot_performance_comparison('e2e', 'p90')

# 生成所有组合的图
print("\n生成所有性能对比图...")
for metric_key in METRICS.keys():
    for quantile_key in QUANTILE_LABELS:
        plot_performance_comparison(metric_key, quantile_key)
        print(f"  ✓ 已生成: {metric_key} @ {quantile_key}")

# ============================================================================
# 综合对比图
# ============================================================================
print("\n绘制综合对比图...")

if auto_quantiles is not None and fixed_quantiles is not None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Performance Comparison: Autoscaling vs Fixed (2P+2T)', 
                 fontsize=16, fontweight='bold')
    
    for idx, (metric_key, metric_info) in enumerate(METRICS.items()):
        ax = axes[idx]
        metric_label = metric_info['label']
        
        # p50, p90, p99 的值
        quantile_labels = ['p50', 'p90', 'p99']
        x = np.arange(len(quantile_labels))
        width = 0.35
        
        auto_values = [auto_quantiles[metric_key][q] for q in quantile_labels]
        fixed_values = [fixed_quantiles[metric_key][q] for q in quantile_labels]
        
        bars1 = ax.bar(x - width/2, auto_values, width, label='Autoscaling', 
                       color='#2ca02c', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, fixed_values, width, label='Fixed (2P+2T)', 
                       color='#ff7f0e', alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel(f'{metric_label} Slowdown', fontweight='bold')
        ax.set_title(f'{metric_label} Slowdown', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([q.upper() for q in quantile_labels])
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = f"{plots_dir}/comprehensive_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 已保存: {save_path}")
    plt.close()

# ============================================================================
# 机器使用时间对比
# ============================================================================
print("\n绘制机器使用时间对比...")

def calculate_machine_hours(df, is_autoscaling=True):
    """
    计算机器使用时间（Machine-Hours）
    
    Args:
        df: DataFrame with time and total_instances columns
        is_autoscaling: 是否是自动扩缩容配置
    
    Returns:
        float: 总机器使用时间（实例数 × 小时）
    """
    if df is None or len(df) == 0:
        return 0.0
    
    total_machine_seconds = 0.0
    
    # 按时间排序
    df_sorted = df.sort_values('time').reset_index(drop=True)
    
    for i in range(len(df_sorted) - 1):
        instances = df_sorted.loc[i, 'total_instances']
        time_start = df_sorted.loc[i, 'time']
        time_end = df_sorted.loc[i + 1, 'time']
        duration = time_end - time_start
        
        total_machine_seconds += instances * duration
    
    # 最后一个时间段到 MAX_TIME（如果设置了）或最后记录的时间
    last_instances = df_sorted.loc[len(df_sorted) - 1, 'total_instances']
    last_time = df_sorted.loc[len(df_sorted) - 1, 'time']
    if MAX_TIME is not None and last_time < MAX_TIME:
        total_machine_seconds += last_instances * (MAX_TIME - last_time)
    
    # 转换为小时
    total_machine_hours = total_machine_seconds / 3600.0
    
    return total_machine_hours

# 计算机器使用时间
auto_machine_hours = calculate_machine_hours(autoscaling_df, is_autoscaling=True)
fixed_10_10_machine_hours = calculate_machine_hours(fixed_8_4_df, is_autoscaling=False)

print(f"  自动扩缩容机器使用时间: {auto_machine_hours:.2f} 机器小时")
print(f"  固定配置(10P+10T)机器使用时间: {fixed_10_10_machine_hours:.2f} 机器小时")

# 绘制机器使用时间对比图
fig, ax = plt.subplots(figsize=(12, 6))

configs = ['Autoscaling\n(5P+5T→Dynamic)', 'Fixed\n(10P+10T)']
machine_hours = [auto_machine_hours, fixed_10_10_machine_hours]
colors = ['#2ca02c', '#ff7f0e']

bars = ax.bar(configs, machine_hours, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# 添加数值标签
for bar, value in zip(bars, machine_hours):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.1f}\nmachine-hours',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# 添加节省百分比标注
if auto_machine_hours > 0:
    saving_vs_10_10 = (fixed_10_10_machine_hours - auto_machine_hours) / fixed_10_10_machine_hours * 100

    # 在自动扩缩容柱子上方添加节省信息
    # if saving_vs_10_10 > 0:
    #     ax.text(0, auto_machine_hours * 0.5,
    #             f'Save {saving_vs_10_10:.1f}%\nvs Fixed(10P+10T)',
    #             ha='center', va='center', fontsize=10, fontweight='bold',
    #             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

ax.set_ylabel('Total Machine-Hours', fontsize=13, fontweight='bold')
ax.set_title('Machine Usage Comparison: Autoscaling vs Fixed Configurations',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

# 设置y轴范围
ax.set_ylim(bottom=0, top=max(machine_hours) * 1.3)

plt.tight_layout()
save_path = f"{plots_dir}/machine_hours_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"  ✓ 已保存: {save_path}")
plt.close()

# ============================================================================
# 统计摘要
# ============================================================================
print("\n" + "=" * 70)
print("统计摘要")
print("=" * 70)

if autoscaling_df is not None:
    print("\n自动扩缩容 (5P+5T → Dynamic):")
    print(f"  最大实例数: {autoscaling_df['total_instances'].max()}")
    print(f"  最小实例数: {autoscaling_df['total_instances'].min()}")
    print(f"  平均实例数: {autoscaling_df['total_instances'].mean():.2f}")
    print(f"  最大 Prompt 实例: {autoscaling_df['prompt_instances'].max()}")
    print(f"  最大 Token 实例: {autoscaling_df['token_instances'].max()}")
    if 'pending_queue_length' in autoscaling_df.columns:
        print(f"  最大队列长度: {autoscaling_df['pending_queue_length'].max()}")
        print(f"  平均队列长度: {autoscaling_df['pending_queue_length'].mean():.1f}")
    if 'total_tps' in autoscaling_df.columns:
        print(f"  最大总 TPS: {autoscaling_df['total_tps'].max():.1f}")
        print(f"  平均总 TPS: {autoscaling_df['total_tps'].mean():.1f}")
        print(f"  平均 Prefill TPS: {autoscaling_df['prefill_tps'].mean():.1f}")
        print(f"  平均 Decode TPS: {autoscaling_df['decode_tps'].mean():.1f}")
    print(f"  机器使用时间: {auto_machine_hours:.2f} machine-hours")

print("\n固定配置 (10P+10T):")
print(f"  总实例数: 20")
print(f"  Prompt 实例: 10")
print(f"  Token 实例: 10")
print(f"  机器使用时间: {fixed_10_10_machine_hours:.2f} machine-hours")


if auto_quantiles is not None and fixed_quantiles is not None:
    print("\n性能对比 (Autoscaling vs Fixed 2P+2T):")
    for metric_key, metric_info in METRICS.items():
        metric_label = metric_info['label']
        print(f"\n  {metric_label} Slowdown:")
        for q in QUANTILE_LABELS:
            auto_val = auto_quantiles[metric_key][q]
            fixed_val = fixed_quantiles[metric_key][q]
            improvement = (fixed_val - auto_val) / fixed_val * 100
            print(f"    {q.upper()}: Auto={auto_val:.3f}, Fixed={fixed_val:.3f}, "
                  f"Improvement={improvement:+.1f}%")

print("\n资源使用对比:")
if auto_machine_hours > 0 and fixed_10_10_machine_hours > 0:
    saving_10_10 = (fixed_10_10_machine_hours - auto_machine_hours) / fixed_10_10_machine_hours * 100
    print(f"  Autoscaling vs Fixed(10P+10T): 节省 {saving_10_10:.1f}% 机器使用时间")

print("\n" + "=" * 70)
print(f"所有图表已保存到: {plots_dir}/")
print("=" * 70)
print("\n✓ 脚本执行完成!")

