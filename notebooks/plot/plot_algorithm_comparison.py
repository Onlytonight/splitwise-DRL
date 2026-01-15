"""
高级科研风格的扩缩容算法对比绘图脚本
对比指标：TTFT, TBT, E2E, 显存利用率, 队列排队时间, 成本, Reward
每个指标单独一个图
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path

sys.path.append('../../notebooks')
from perf_model import PerfModel
from utils import *

# 设置matplotlib全局参数 - 科研论文风格
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'grid.alpha': 0.3,
    'axes.grid': True,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
})

# 配置
RESULTS_DIR = "../../results"
PLOTS_DIR = "../../plots"
PERF_MODEL_PATH = "../../data/perf_model.csv"

# 配色方案 - 科研论文常用配色
COLORS = {
    'adaptive_pool': '#2E86AB',      # 深蓝
    'mixed_pool': '#A23B72',         # 紫红
    'baseline_heteroscale': '#F18F01', # 橙色
    'baseline_utilization': '#C73E1D', # 红色
    'baseline_queue': '#6A994E',     # 绿色
    'rl_sac': '#BC4B51',            # 暗红
}

# 标记样式
MARKERS = {
    'adaptive_pool': 'o',
    'mixed_pool': 's',
    'baseline_heteroscale': '^',
    'baseline_utilization': 'D',
    'baseline_queue': 'v',
    'rl_sac': 'p',
}

# 线条样式
LINESTYLES = {
    'adaptive_pool': '-',
    'mixed_pool': '-',
    'baseline_heteroscale': '--',
    'baseline_utilization': '-.',
    'baseline_queue': ':',
    'rl_sac': '--',
}


def get_summary_data(results_dir, scheduler, start_state, cluster, trace, seed, interval, model="", trace_epoch=None):
    """
    读取summary.csv文件
    
    Args:
        trace_epoch: 对于RL方法，指定读取第几个trace的summary（None表示读取最新的）
    """
    try:
        # 路径格式：results/{seed}/{start_state}/{trace}/{model}/mixed_pool/{scheduler}
        base_path = f"{results_dir}/{seed}/{start_state}/{trace}/{model}/mixed_pool/{scheduler}"
        if interval != 0:
            base_path = f"{base_path}/{interval}"
        
        # 检查是否存在summary.csv（baseline方法）
        summary_path = f"{base_path}/summary.csv"
        if os.path.exists(summary_path):
            return pd.read_csv(summary_path)
        
        # 检查是否存在summary_trace_XXX.csv（RL方法）
        import glob
        trace_summaries = glob.glob(f"{base_path}/summary_trace_*.csv")
        if trace_summaries:
            # 提取trace编号并排序
            trace_numbers = []
            for f in trace_summaries:
                try:
                    num = int(f.split('_trace_')[-1].replace('.csv', ''))
                    trace_numbers.append((num, f))
                except:
                    continue
            
            if trace_numbers:
                trace_numbers.sort(key=lambda x: x[0])
                # 使用最后一个trace（训练最充分的）
                if trace_epoch is None:
                    selected_file = trace_numbers[-1][1]
                    print(f"  使用RL trace {trace_numbers[-1][0]}: {os.path.basename(selected_file)}")
                else:
                    # 查找指定的trace_epoch
                    for num, file in trace_numbers:
                        if num == trace_epoch:
                            selected_file = file
                            break
                    else:
                        selected_file = trace_numbers[-1][1]
                
                return pd.read_csv(selected_file)
        
        print(f"Failed to find summary file in: {base_path}")
        return None
    except Exception as e:
        print(f"Failed to read summary data from {base_path}")
        print(f"Error: {e}")
        return None


def get_reward_data(results_dir, scheduler,reward_tag, start_state, cluster, trace, seed, interval, model=""):
    """读取reward.csv文件"""
    try:
        # 路径格式：results/{seed}/{start_state}/{trace}/{model}/mixed_pool/{scheduler}
        base_path = f"{results_dir}/{seed}/{start_state}/{trace}/{model}/mixed_pool/{scheduler}"
        if interval != 0:
            base_path = f"{base_path}/{interval}"
        
        # 尝试不同的reward文件名
        # 先尝试 reward_{scheduler}.csv
        path = f"{base_path}/reward_{reward_tag}.csv"
        if os.path.exists(path):
            reward_df = pd.read_csv(path)
            return reward_df
        
        # 再尝试 reward.csv
        path = f"{base_path}/reward.csv"
        if os.path.exists(path):
            reward_df = pd.read_csv(path)
            return reward_df
        
        print(f"  Reward file not found for {scheduler}")
        return None
    except Exception as e:
        print(f"Failed to read reward.csv from {base_path}")
        print(f"Error: {e}")
        return None


def get_request_data(results_dir, scheduler, start_state, cluster, trace, seed, interval, model="", trace_epoch=None):
    """
    读取详细请求数据
    
    Args:
        trace_epoch: 对于RL方法，指定读取第几个trace的详细数据（None表示读取最新的）
    """
    try:
        # 路径格式：results/{seed}/{start_state}/{trace}/{model}/mixed_pool/{scheduler}/detailed
        base_path = f"{results_dir}/{seed}/{start_state}/{trace}/{model}/mixed_pool/{scheduler}/detailed"
        if interval != 0:
            base_path = f"{results_dir}/{seed}/{start_state}/{trace}/{model}/mixed_pool/{scheduler}/{interval}/detailed"
        
        # 检查是否存在0.csv（baseline方法）
        detail_path = f"{base_path}/0.csv"
        if os.path.exists(detail_path):
            return pd.read_csv(detail_path)
        
        # 检查是否存在0_trace_XXX.csv（RL方法）
        import glob
        trace_details = glob.glob(f"{base_path}/0_trace_*.csv")
        if trace_details:
            # 提取trace编号并排序
            trace_numbers = []
            for f in trace_details:
                try:
                    num = int(f.split('_trace_')[-1].replace('.csv', ''))
                    trace_numbers.append((num, f))
                except:
                    continue
            
            if trace_numbers:
                trace_numbers.sort(key=lambda x: x[0])
                # 使用最后一个trace（训练最充分的）
                if trace_epoch is None:
                    selected_file = trace_numbers[-1][1]
                else:
                    # 查找指定的trace_epoch
                    for num, file in trace_numbers:
                        if num == trace_epoch:
                            selected_file = file
                            break
                    else:
                        selected_file = trace_numbers[-1][1]
                
                return pd.read_csv(selected_file)
        
        print(f"Failed to find detailed request file in: {base_path}")
        return None
    except Exception as e:
        print(f"Failed to read detailed request data from {base_path}")
        print(f"Error: {e}")
        return None


def collect_all_metrics(configs, traces, seed=0, model="bloom-176b", quantiles=[0.5, 0.9, 0.99], trace_epoch=None):
    """
    收集所有算法在不同traces下的性能指标
    返回一个包含所有指标的DataFrame
    
    Args:
        trace_epoch: 对于RL方法，指定读取第几个trace的数据（None表示读取最新的）
    """
    results = []
    
    for trace in traces:
        for config in configs:
            name = config["name"]
            scheduler = config["scheduler"]
            reward_tag= config.get("reward_tag", "")
            start_state = config["start_state"]
            cluster = config["cluster"]
            interval = config.get("interval", 0)
            
            # 读取summary数据
            summary_df = get_summary_data(RESULTS_DIR, scheduler, start_state, cluster, trace, seed, interval, model, trace_epoch)
            if summary_df is None:
                continue
            
            # 读取reward数据
            reward_df = get_reward_data(RESULTS_DIR,scheduler, reward_tag, start_state, cluster, trace, seed, interval, model)
            
            # 读取请求详细数据用于计算slowdown
            request_df = get_request_data(RESULTS_DIR, scheduler, start_state, cluster, trace, seed, interval, model, trace_epoch)
            
            result = {
                'name': name,
                'scheduler': scheduler,
                'trace': trace,
                'seed': seed,
                'cluster': cluster,
            }
            
            # 提取TTFT指标 (各分位数)
            for q in quantiles:
                q_int = int(q * 100)
                result[f'ttft_p{q_int}'] = summary_df[f'ttft_times_p{q_int}'].iloc[0]
                result[f'tbt_p{q_int}'] = summary_df[f'tbt_times_p{q_int}'].iloc[0]
                result[f'e2e_p{q_int}'] = summary_df[f'response_times_p{q_int}'].iloc[0]
                result[f'queue_time_p{q_int}'] = summary_df[f'queue_times_p{q_int}'].iloc[0]
            
            # 平均值
            result['ttft_mean'] = summary_df['ttft_times_mean'].iloc[0]
            result['tbt_mean'] = summary_df['tbt_times_mean'].iloc[0]
            result['e2e_mean'] = summary_df['response_times_mean'].iloc[0]
            result['queue_time_mean'] = summary_df['queue_times_mean'].iloc[0]
            
            # 如果有reward数据，提取相关指标
            if reward_df is not None and len(reward_df) > 0:
                result['avg_reward'] = reward_df['reward'].mean()
                result['avg_cost'] = reward_df['raw_cost'].mean() if 'raw_cost' in reward_df.columns else reward_df['cost_penalty'].abs().mean()
                result['avg_util'] = reward_df['util_avg'].mean() if 'util_avg' in reward_df.columns else 0
            else:
                result['avg_reward'] = 0
                result['avg_cost'] = 0
                result['avg_util'] = 0
            
            # 计算slowdown（如果有baseline数据）
            if request_df is not None:
                perf_model = PerfModel(PERF_MODEL_PATH, init=True)
                perf_model.add_baseline_perf(request_df, model, "h100-80gb", 8)
                request_df["baseline_e2e"] = request_df["baseline_ttft"] + request_df["baseline_tbt"] * (request_df["token_sizes"] - 1)
                
                for q in quantiles:
                    q_int = int(q * 100)
                    result[f'ttft_slowdown_p{q_int}'] = (request_df["ttft_times"] / request_df["baseline_ttft"]).quantile(q)
                    result[f'tbt_slowdown_p{q_int}'] = (request_df["tbt_times"] / request_df["baseline_tbt"]).quantile(q)
                    result[f'e2e_slowdown_p{q_int}'] = (request_df["response_times"] / request_df["baseline_e2e"]).quantile(q)
            
            results.append(result)
    
    return pd.DataFrame(results)


def plot_single_metric(df, traces, metric_name, ylabel, filename, 
                       quantile=None, ylim=None, slo=None, 
                       show_legend=True, legend_loc='best'):
    """
    绘制单个指标的对比图 - 科研风格
    
    Args:
        df: 数据DataFrame
        traces: trace列表
        metric_name: 指标列名
        ylabel: Y轴标签
        filename: 保存文件名
        quantile: 分位数（用于标题）
        ylim: Y轴范围 (ymin, ymax)
        slo: SLO阈值线
        show_legend: 是否显示图例
        legend_loc: 图例位置
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 提取trace的负载数值（假设格式为 rr_conv_10 -> 10）
    x_values = [int(t.split('_')[-1]) for t in traces]
    
    # 为每个算法绘制线条
    for name in df['name'].unique():
        subset = df[df['name'] == name].sort_values('trace')
        
        # 按trace顺序排列
        y_values = []
        for trace in traces:
            trace_data = subset[subset['trace'] == trace]
            if len(trace_data) > 0:
                y_values.append(trace_data[metric_name].iloc[0])
            else:
                y_values.append(np.nan)
        
        # 绘制线条和标记
        ax.plot(x_values, y_values, 
                label=name,
                color=COLORS.get(name, '#000000'),
                marker=MARKERS.get(name, 'o'),
                linestyle=LINESTYLES.get(name, '-'),
                linewidth=2,
                markersize=8,
                markeredgewidth=0.5,
                markeredgecolor='white',
                alpha=0.9)
    
    # 添加SLO阈值线
    if slo is not None:
        ax.axhline(y=slo, color='red', linestyle='--', linewidth=1.5, 
                   label=f'SLO Threshold', alpha=0.7, zorder=0)
    
    # 设置标签和标题
    ax.set_xlabel('Request Rate (req/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    
    if quantile:
        title = f"{ylabel} (P{int(quantile*100)})"
    else:
        title = ylabel
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # 设置Y轴范围
    if ylim:
        ax.set_ylim(ylim)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 图例
    if show_legend:
        ax.legend(loc=legend_loc, frameon=True, shadow=True, 
                 fancybox=True, framealpha=0.95, edgecolor='gray')
    
    # 紧凑布局
    plt.tight_layout()
    
    # 保存
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ 已保存: {save_path}")
    
    plt.close()


def smooth_data(data, window_size=50):
    """
    使用滑动窗口平滑数据
    
    Args:
        data: 原始数据
        window_size: 窗口大小
    
    Returns:
        平滑后的数据
    """
    if len(data) < window_size:
        return data
    
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    # 为了保持长度一致，前面填充原始数据
    padding = data[:window_size-1]
    return np.concatenate([padding, smoothed])


def plot_reward_curves(configs, seed=1, model="bloom-176b", start_state="splitwise_22_10", 
                       cluster="0_44", window_size=50):
    """
    绘制奖励曲线对比图
    
    Args:
        configs: 算法配置列表
        seed: 随机种子
        model: 模型名称
        start_state: 起始状态
        cluster: 集群配置
        window_size: 平滑窗口大小
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 定义要绘制的指标
    metrics_to_plot = [
        ('reward', 'Total Reward', 0),
        ('cost_penalty', 'Cost Penalty', 1),
        ('blocking_penalty', 'Blocking Penalty', 2),
        ('saturation_penalty', 'Saturation Penalty', 3),
    ]
    
    for config in configs:
        name = config["name"]
        if "RL" not in name:
            continue
        scheduler = config["scheduler"]
        reward_tag = config.get("reward_tag", scheduler)
        
        # 读取reward文件
        base_path = f"{RESULTS_DIR}/{seed}/{start_state}/{cluster}/{model}/mixed_pool/{scheduler}"
        reward_path = f"{base_path}/reward_{reward_tag}.csv"
        
        if not os.path.exists(reward_path):
            print(f"⚠ Reward文件不存在: {reward_path}")
            continue
        
        print(f"✓ 读取 {name} 的奖励数据: {reward_path}")
        reward_df = pd.read_csv(reward_path)
        
        # 绘制各个指标
        for metric_name, metric_label, ax_idx in metrics_to_plot:
            if metric_name not in reward_df.columns:
                continue
            
            ax = axes[ax_idx]
            
            # 原始数据
            steps = reward_df['step'].values
            values = reward_df[metric_name].values
            
            # 平滑数据
            smoothed_values = smooth_data(values, window_size=window_size)
            
            # 绘制平滑曲线（主线）
            ax.plot(steps, smoothed_values,
                   label=f'{name}',
                   color=COLORS.get(name, '#2E86AB'),
                   linestyle=LINESTYLES.get(name, '-'),
                   linewidth=2.5,
                   alpha=0.9)
            
            # 绘制原始数据（淡化的背景线）
            ax.plot(steps, values,
                   color=COLORS.get(name, '#2E86AB'),
                   linewidth=0.5,
                   alpha=0.2)
    
    # 设置每个子图的属性
    for metric_name, metric_label, ax_idx in metrics_to_plot:
        ax = axes[ax_idx]
        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_label} (Window={window_size})', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.legend(loc='best', frameon=True, shadow=True, fancybox=True, 
                 framealpha=0.95, edgecolor='gray')
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, 'comparison_reward_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"✓ 已保存奖励曲线图: {save_path}")
    
    plt.close()


def plot_resource_curves(configs, seed=1, model="bloom-176b", start_state="splitwise_22_10",
                         cluster="0_44", window_size=50):
    """
    绘制资源使用曲线（实例数、利用率）
    
    Args:
        configs: 算法配置列表
        seed: 随机种子
        model: 模型名称
        start_state: 起始状态
        cluster: 集群配置
        window_size: 平滑窗口大小
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 定义要绘制的指标
    metrics_to_plot = [
        ('n_p', 'Prompt Instances', 0),
        ('n_t', 'Token Instances', 1),
        ('util_mem_p', 'Prompt Memory Utilization', 2),
        ('util_mem_t', 'Token Memory Utilization', 3),
    ]
    
    for config in configs:
        name = config["name"]
        scheduler = config["scheduler"]
        reward_tag = config.get("reward_tag", scheduler)
        
        # 读取reward文件
        base_path = f"{RESULTS_DIR}/{seed}/{start_state}/{cluster}/{model}/mixed_pool/{scheduler}"
        reward_path = f"{base_path}/reward_{reward_tag}.csv"
        
        if not os.path.exists(reward_path):
            continue
        
        reward_df = pd.read_csv(reward_path)
        
        # 绘制各个指标
        for metric_name, metric_label, ax_idx in metrics_to_plot:
            if metric_name not in reward_df.columns:
                continue
            
            ax = axes[ax_idx]
            
            # 原始数据
            steps = reward_df['step'].values
            values = reward_df[metric_name].values
            
            # 平滑数据
            smoothed_values = smooth_data(values, window_size=window_size)
            
            # 绘制平滑曲线
            ax.plot(steps, smoothed_values,
                   label=f'{name}',
                   color=COLORS.get(name, '#2E86AB'),
                   linestyle=LINESTYLES.get(name, '-'),
                   linewidth=2.5,
                   alpha=0.9)
            
            # 绘制原始数据（淡化）
            ax.plot(steps, values,
                   color=COLORS.get(name, '#2E86AB'),
                   linewidth=0.5,
                   alpha=0.2)
    
    # 设置每个子图的属性
    for metric_name, metric_label, ax_idx in metrics_to_plot:
        ax = axes[ax_idx]
        ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_label} (Window={window_size})', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.legend(loc='best', frameon=True, shadow=True, fancybox=True,
                 framealpha=0.95, edgecolor='gray')
        
        # 对于利用率指标，设置Y轴范围为[0, 1]
        if 'util' in metric_name.lower():
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(PLOTS_DIR, exist_ok=True)
    save_path = os.path.join(PLOTS_DIR, 'comparison_resource_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"✓ 已保存资源使用曲线图: {save_path}")
    
    plt.close()


def plot_bar_comparison(df, quantile=0.99):
    """
    绘制柱状图对比（用于单trace情况）
    
    Args:
        df: 包含所有指标的DataFrame
        quantile: 使用的分位数
    """
    q_int = int(quantile * 100)
    
    # 定义所有要绘制的指标
    metrics = [
        (f'ttft_p{q_int}', f'TTFT P{q_int} (s)', None, None),
        (f'tbt_p{q_int}', f'TBT P{q_int} (s)', None, None),
        (f'e2e_p{q_int}', f'E2E Latency P{q_int} (s)', None, None),
        (f'queue_time_p{q_int}', f'Queue Time P{q_int} (s)', None, None),
        ('avg_util', 'Resource Utilization', (0, 1), None),
        ('avg_cost', 'Average Cost (instances)', None, None),
        ('avg_reward', 'Average Reward', None, None),
    ]
    
    # 添加slowdown指标（如果存在）
    if f'ttft_slowdown_p{q_int}' in df.columns:
        metrics.append((f'ttft_slowdown_p{q_int}', f'TTFT Slowdown P{q_int}', (0, 8), 6 if quantile == 0.99 else 3))
    if f'tbt_slowdown_p{q_int}' in df.columns:
        metrics.append((f'tbt_slowdown_p{q_int}', f'TBT Slowdown P{q_int}', (0, 8), 5 if quantile == 0.99 else 1.5))
    if f'e2e_slowdown_p{q_int}' in df.columns:
        metrics.append((f'e2e_slowdown_p{q_int}', f'E2E Slowdown P{q_int}', (0, 8), 5 if quantile == 0.99 else 1.5))
    
    for metric_name, ylabel, ylim, slo in metrics:
        if metric_name not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # 提取数据
        algorithms = df['name'].tolist()
        values = df[metric_name].tolist()
        
        # 设置柱状图位置
        x_pos = np.arange(len(algorithms))
        
        # 绘制柱状图
        bars = ax.bar(x_pos, values, width=0.6, alpha=0.85, edgecolor='black', linewidth=1.2)
        
        # 为每个算法设置不同的颜色
        for i, (bar, alg) in enumerate(zip(bars, algorithms)):
            bar.set_color(COLORS.get(alg, '#2E86AB'))
        
        # 在柱子上方显示数值
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}' if val < 10 else f'{val:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 添加SLO阈值线
        if slo is not None:
            ax.axhline(y=slo, color='red', linestyle='--', linewidth=2, 
                      label='SLO Threshold', alpha=0.7, zorder=0)
            ax.legend(loc='upper right')
        
        # 设置标签
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_title(ylabel, fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algorithms, fontsize=11)
        
        # 设置Y轴范围
        if ylim:
            ax.set_ylim(ylim)
        else:
            # 自动设置，留出空间显示数值
            ax.set_ylim(bottom=0, top=max(values) * 1.15)
        
        # 网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
        ax.set_axisbelow(True)
        
        # 紧凑布局
        plt.tight_layout()
        
        # 保存
        os.makedirs(PLOTS_DIR, exist_ok=True)
        filename = f'comparison_bar_{metric_name}.png'
        save_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"✓ 已保存: {save_path}")
        
        plt.close()


def plot_all_metrics_comparison(df, traces, quantile=0.99):
    """
    生成所有指标的对比图
    
    Args:
        df: 包含所有指标的DataFrame
        traces: trace列表
        quantile: 使用的分位数
    """
    q_int = int(quantile * 100)
    
    # 1. TTFT对比
    plot_single_metric(
        df, traces,
        metric_name=f'ttft_p{q_int}',
        ylabel='Time to First Token (s)',
        filename=f'comparison_ttft_p{q_int}.png',
        quantile=quantile,
        ylim=(0, None),
        slo=None,
    )
    
    # 2. TBT对比
    plot_single_metric(
        df, traces,
        metric_name=f'tbt_p{q_int}',
        ylabel='Time Between Tokens (s)',
        filename=f'comparison_tbt_p{q_int}.png',
        quantile=quantile,
        ylim=(0, None),
        slo=None,
    )
    
    # 3. E2E延迟对比
    plot_single_metric(
        df, traces,
        metric_name=f'e2e_p{q_int}',
        ylabel='End-to-End Latency (s)',
        filename=f'comparison_e2e_p{q_int}.png',
        quantile=quantile,
        ylim=(0, None),
        slo=None,
    )
    
    # 4. 队列排队时间对比
    plot_single_metric(
        df, traces,
        metric_name=f'queue_time_p{q_int}',
        ylabel='Queue Waiting Time (s)',
        filename=f'comparison_queue_time_p{q_int}.png',
        quantile=quantile,
        ylim=(0, None),
        slo=None,
    )
    
    # 5. 显存利用率对比
    plot_single_metric(
        df, traces,
        metric_name='avg_util',
        ylabel='Average Resource Utilization',
        filename='comparison_utilization.png',
        ylim=(0, 1),
        slo=None,
    )
    
    # 6. 成本对比
    plot_single_metric(
        df, traces,
        metric_name='avg_cost',
        ylabel='Average Cost (instances)',
        filename='comparison_cost.png',
        ylim=(0, None),
        slo=None,
    )
    
    # 7. Reward对比
    plot_single_metric(
        df, traces,
        metric_name='avg_reward',
        ylabel='Average Reward',
        filename='comparison_reward.png',
        ylim=(None, None),
        slo=None,
    )
    
    # 8. TTFT Slowdown (如果有)
    if f'ttft_slowdown_p{q_int}' in df.columns:
        plot_single_metric(
            df, traces,
            metric_name=f'ttft_slowdown_p{q_int}',
            ylabel='TTFT Slowdown',
            filename=f'comparison_ttft_slowdown_p{q_int}.png',
            quantile=quantile,
            ylim=(0, 8),
            slo=6 if quantile == 0.99 else (3 if quantile == 0.9 else 2),
        )
    
    # 9. TBT Slowdown (如果有)
    if f'tbt_slowdown_p{q_int}' in df.columns:
        plot_single_metric(
            df, traces,
            metric_name=f'tbt_slowdown_p{q_int}',
            ylabel='TBT Slowdown',
            filename=f'comparison_tbt_slowdown_p{q_int}.png',
            quantile=quantile,
            ylim=(0, 8),
            slo=5 if quantile == 0.99 else (1.5 if quantile == 0.9 else 1.25),
        )
    
    # 10. E2E Slowdown (如果有)
    if f'e2e_slowdown_p{q_int}' in df.columns:
        plot_single_metric(
            df, traces,
            metric_name=f'e2e_slowdown_p{q_int}',
            ylabel='E2E Slowdown',
            filename=f'comparison_e2e_slowdown_p{q_int}.png',
            quantile=quantile,
            ylim=(0, 8),
            slo=5 if quantile == 0.99 else (1.5 if quantile == 0.9 else 1.25),
        )


def main():
    """
    主函数：对比不同扩缩容算法的性能
    """
    print("=" * 80)
    print("扩缩容算法性能对比 - 高级科研风格绘图")
    print("对比算法: RL SAC vs Baseline HeteroScale")
    print("=" * 80)
    
    # 定义要对比的算法配置
    # 路径格式：results/{seed}/{start_state}/{trace}/{model}/mixed_pool/{scheduler}
    configs = [
        {
            "name": "RL-SAC",
            "scheduler": "rl_sac",
            "reward_tag":"sac",
            "start_state": "splitwise_22_10",
            "cluster": "0_45",  # 这里cluster参数不再使用，但保留以兼容
            "interval": 0,
        },
        {
            "name": "HeteroScale",
            "scheduler": "baseline_heteroscale",
            "reward_tag":"heteroscale",
            "start_state": "splitwise_22_10",
            "cluster": "0_44",
            "interval": 0,
        },
    ]
    
    # 定义要分析的traces
    # 如果你有多个trace，可以在这里列出；目前只有0_44和test_trace
    traces = ["0_44"]  # 可以添加 "test_trace" 如果需要对比
    
    # 收集所有指标数据
    print("\n正在收集数据...")
    print(f"  Seed: 1")
    print(f"  Model: bloom-176b")
    print(f"  Traces: {traces}")
    
    # 对于RL方法，使用最后一个训练epoch (trace_epoch=None会自动选择最新的)
    df = collect_all_metrics(configs, traces, seed=1, model="bloom-176b", trace_epoch=300)
    
    if len(df) == 0:
        print("❌ 未能收集到数据，请检查路径配置")
        return
    
    print(f"✓ 数据收集完成，共 {len(df)} 条记录")
    print(f"  算法: {df['name'].unique().tolist()}")
    
    # 1. 绘制奖励曲线（训练过程）
    print("\n正在生成奖励曲线图...")
    plot_reward_curves(
        configs, 
        seed=1, 
        model="bloom-176b",
        start_state="splitwise_22_10",
        cluster="0_44",
        window_size=1000  # 可以调整窗口大小
    )
    
    # 2. 绘制资源使用曲线
    print("\n正在生成资源使用曲线图...")
    plot_resource_curves(
        configs,
        seed=1,
        model="bloom-176b", 
        start_state="splitwise_22_10",
        cluster="0_44",
        window_size=50
    )
    
    # 3. 绘制性能对比图
    # 如果只有一个trace，不需要绘制vs trace的图，而是绘制对比柱状图
    if len(traces) == 1:
        print("\n正在生成算法性能对比图（柱状图）...")
        plot_bar_comparison(df, quantile=0.99)
    else:
        # 如果有多个trace，绘制折线图
        print("\n正在生成算法性能对比图（折线图）...")
        plot_all_metrics_comparison(df, traces, quantile=0.99)
    
    print("\n" + "=" * 80)
    print("✓ 所有图表生成完成！")
    print(f"  输出目录: {PLOTS_DIR}")
    print("  生成的图表:")
    print("    - comparison_reward_curves.png (奖励曲线)")
    print("    - comparison_resource_curves.png (资源使用曲线)")
    print("    - comparison_bar_*.png (性能对比柱状图)")
    print("=" * 80)


if __name__ == "__main__":
    main()
