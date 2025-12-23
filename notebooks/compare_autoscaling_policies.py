"""
对比不同自动扩缩容策略的性能
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import get_summary_data, get_request_data
from perf_model import PerfModel

# 设置全局字体大小
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})

# 设置 seaborn 样式
sns.set_style("whitegrid")
sns.set_palette("husl")

results_dir = "../results"
plots_dir = "../plots/autoscaling_comparison/"
perf_model_path = "../data/perf_model.csv"
os.makedirs(plots_dir, exist_ok=True)


def get_autoscaling_data(policies, traces, seed, start_state, cluster, scheduler, model="bloom-176b", quantiles=[0.5, 0.9, 0.99]):
    """
    加载不同自动扩缩容策略的数据
    
    Args:
        policies: 扩缩容策略列表，如 ["heteroscale", "hpa_gpu", "independent_tps"]
        traces: 流量追踪列表
        seed: 随机种子
        start_state: 启动状态
        cluster: 集群配置
        scheduler: 调度器类型
        model: 模型名称
        quantiles: 分位数列表
    
    Returns:
        results_df: 结果数据框
        request_dfs: 请求数据字典
    """
    results = []
    request_dfs = {}
    
    for trace in traces:
        for policy in policies:
            # 构建路径：results/seed/start_state/trace/cluster/model/scheduler/policy/
            result_path = f"{results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/{policy}"
            
            if not os.path.exists(result_path):
                print(f"警告: 路径不存在: {result_path}")
                continue
            
            # 读取数据
            summary_df = None
            request_df = None
            
            try:
                # 尝试读取 summary.csv 和 request.csv
                summary_path = f"{result_path}/summary.csv"
                request_path = f"{result_path}/request.csv"
                
                if os.path.exists(summary_path):
                    summary_df = pd.read_csv(summary_path)
                if os.path.exists(request_path):
                    request_df = pd.read_csv(request_path)
                
                if summary_df is None or request_df is None:
                    print(f"警告: 数据文件缺失: {result_path}")
                    continue
                
                # 计算归一化指标
                perf_model = PerfModel(perf_model_path, init=True)
                normalize_hardware = "a100-80gb"
                normalize_tp = 8
                
                perf_model.add_baseline_perf(request_df, model, normalize_hardware, normalize_tp)
                request_df["baseline_e2e"] = request_df["baseline_ttft"] + request_df["baseline_tbt"] * (request_df["token_sizes"] - 1)
                request_df["ttft_slowdown"] = request_df["ttft_times"] / request_df["baseline_ttft"]
                request_df["tbt_slowdown"] = request_df["tbt_times"] / request_df["baseline_tbt"]
                request_df["e2e_slowdown"] = request_df["response_times"] / request_df["baseline_e2e"]
                
                # 检查是否 OOM
                oom = os.path.exists(f"{result_path}/oom.csv")
                
                # 构建结果字典
                result = {
                    "policy": policy,
                    "trace": trace,
                    "seed": seed,
                    "oom": oom
                }
                
                # 添加分位数数据
                for quantile in quantiles:
                    result[f"ttft_slowdown_p{int(quantile * 100)}"] = request_df["ttft_slowdown"].quantile(quantile)
                    result[f"tbt_slowdown_p{int(quantile * 100)}"] = request_df["tbt_slowdown"].quantile(quantile)
                    result[f"e2e_slowdown_p{int(quantile * 100)}"] = request_df["e2e_slowdown"].quantile(quantile)
                    
                    result[f"ttft_times_p{int(quantile * 100)}"] = request_df["ttft_times"].quantile(quantile)
                    result[f"tbt_times_p{int(quantile * 100)}"] = request_df["tbt_times"].quantile(quantile)
                    result[f"e2e_times_p{int(quantile * 100)}"] = request_df["response_times"].quantile(quantile)
                
                # 添加吞吐量和其他指标
                result["avg_throughput"] = len(request_df) / request_df["response_times"].max() if len(request_df) > 0 else 0
                result["total_requests"] = len(request_df)
                
                results.append(result)
                request_dfs[f"{policy}_{trace}"] = request_df
                
                print(f"✓ 加载成功: {policy} - {trace}")
                
            except Exception as e:
                print(f"错误: 无法处理 {result_path}: {e}")
                continue
    
    results_df = pd.DataFrame(results)
    return results_df, request_dfs


def plot_policy_comparison(results_df, 
                          traces,
                          metric="ttft_slowdown",
                          quantiles=[0.5, 0.9, 0.99],
                          title=None,
                          save_path=None):
    """
    绘制不同策略在不同分位数下的对比图
    
    Args:
        results_df: 结果数据框
        traces: 流量追踪列表
        metric: 指标名称 (ttft_slowdown, tbt_slowdown, e2e_slowdown)
        quantiles: 分位数列表
        title: 图表标题
        save_path: 保存路径
    """
    fig, axs = plt.subplots(1, len(quantiles), figsize=(len(quantiles) * 4.5, 4), sharey=True, constrained_layout=True)
    
    if len(quantiles) == 1:
        axs = [axs]
    
    # 获取指标的中文名称
    metric_names = {
        "ttft_slowdown": "TTFT Slowdown",
        "tbt_slowdown": "TBT Slowdown",
        "e2e_slowdown": "E2E Slowdown",
        "ttft_times": "TTFT Time (s)",
        "tbt_times": "TBT Time (s)",
        "e2e_times": "E2E Time (s)"
    }
    metric_label = metric_names.get(metric, metric)
    
    # 获取 SLO 线
    slo_values = {
        "ttft_slowdown": {0.5: 2.0, 0.9: 3.0, 0.99: 6.0},
        "tbt_slowdown": {0.5: 1.25, 0.9: 1.5, 0.99: 5.0},
        "e2e_slowdown": {0.5: 1.25, 0.9: 1.5, 0.99: 5.0}
    }
    
    for i, quantile in enumerate(quantiles):
        ax = axs[i]
        
        # 绘制折线图
        sns.lineplot(
            data=results_df,
            x="trace",
            y=f"{metric}_p{int(quantile * 100)}",
            hue="policy",
            style="policy",
            markers=True,
            markersize=8,
            linewidth=2,
            ax=ax
        )
        
        # 添加 SLO 线（如果有）
        if metric in slo_values and quantile in slo_values[metric]:
            slo = slo_values[metric][quantile]
            ax.axhline(y=slo, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"SLO={slo}")
        
        # 设置标题和标签
        ax.set_title(f"P{int(quantile * 100)}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Request Rate (req/s)", fontsize=10)
        ax.set_ylabel(metric_label if i == 0 else "", fontsize=10)
        
        # 设置 x 轴标签
        xlabels = [trace.split("_")[-1] for trace in traces]
        ax.set_xticks(range(len(traces)))
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        
        # 网格
        ax.grid(True, alpha=0.3)
        
        # 移除子图的图例
        if ax.get_legend():
            ax.get_legend().remove()
    
    # 创建全局图例
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(results_df['policy'].unique()), 
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.08)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {save_path}")
    plt.close()


def plot_metrics_grid(results_df,
                     traces, 
                     metrics=["ttft_slowdown", "tbt_slowdown", "e2e_slowdown"],
                     quantile=0.9,
                     title=None,
                     save_path=None):
    """
    绘制多个指标的网格对比图（固定分位数）
    
    Args:
        results_df: 结果数据框
        traces: 流量追踪列表
        metrics: 指标列表
        quantile: 分位数
        title: 图表标题
        save_path: 保存路径
    """
    fig, axs = plt.subplots(1, len(metrics), figsize=(len(metrics) * 4.5, 4), sharey=False, constrained_layout=True)
    
    if len(metrics) == 1:
        axs = [axs]
    
    metric_names = {
        "ttft_slowdown": "TTFT Slowdown",
        "tbt_slowdown": "TBT Slowdown",
        "e2e_slowdown": "E2E Slowdown"
    }
    
    slo_values = {
        "ttft_slowdown": {0.5: 2.0, 0.9: 3.0, 0.99: 6.0},
        "tbt_slowdown": {0.5: 1.25, 0.9: 1.5, 0.99: 5.0},
        "e2e_slowdown": {0.5: 1.25, 0.9: 1.5, 0.99: 5.0}
    }
    
    for i, metric in enumerate(metrics):
        ax = axs[i]
        
        # 绘制折线图
        sns.lineplot(
            data=results_df,
            x="trace",
            y=f"{metric}_p{int(quantile * 100)}",
            hue="policy",
            style="policy",
            markers=True,
            markersize=8,
            linewidth=2,
            ax=ax
        )
        
        # 添加 SLO 线
        if metric in slo_values and quantile in slo_values[metric]:
            slo = slo_values[metric][quantile]
            ax.axhline(y=slo, color="red", linestyle="--", linewidth=2, alpha=0.7)
        
        # 设置标题和标签
        metric_label = metric_names.get(metric, metric)
        ax.set_title(f"{metric_label} (P{int(quantile * 100)})", fontsize=12, fontweight='bold')
        ax.set_xlabel("Request Rate (req/s)", fontsize=10)
        ax.set_ylabel(metric_label if i == 0 else "", fontsize=10)
        
        # 设置 x 轴标签
        xlabels = [trace.split("_")[-1] for trace in traces]
        ax.set_xticks(range(len(traces)))
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        
        # 网格
        ax.grid(True, alpha=0.3)
        
        # 移除子图的图例
        if ax.get_legend():
            ax.get_legend().remove()
    
    # 创建全局图例
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(results_df['policy'].unique()), 
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.08)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {save_path}")
    plt.close()


def plot_bar_comparison(results_df,
                       metric="ttft_slowdown",
                       quantile=0.9,
                       aggregate="mean",
                       title=None,
                       save_path=None):
    """
    绘制不同策略的柱状对比图（聚合所有 trace）
    
    Args:
        results_df: 结果数据框
        metric: 指标名称
        quantile: 分位数
        aggregate: 聚合方式 ("mean", "median")
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    
    metric_col = f"{metric}_p{int(quantile * 100)}"
    
    # 按策略聚合
    if aggregate == "mean":
        agg_df = results_df.groupby("policy")[metric_col].mean().reset_index()
    else:
        agg_df = results_df.groupby("policy")[metric_col].median().reset_index()
    
    # 绘制柱状图
    bars = ax.bar(agg_df["policy"], agg_df[metric_col], alpha=0.8, edgecolor='black')
    
    # 为每个柱子添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # 添加 SLO 线
    slo_values = {
        "ttft_slowdown": {0.5: 2.0, 0.9: 3.0, 0.99: 6.0},
        "tbt_slowdown": {0.5: 1.25, 0.9: 1.5, 0.99: 5.0},
        "e2e_slowdown": {0.5: 1.25, 0.9: 1.5, 0.99: 5.0}
    }
    if metric in slo_values and quantile in slo_values[metric]:
        slo = slo_values[metric][quantile]
        ax.axhline(y=slo, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"SLO={slo}")
        ax.legend()
    
    # 设置标签
    metric_names = {
        "ttft_slowdown": "TTFT Slowdown",
        "tbt_slowdown": "TBT Slowdown",
        "e2e_slowdown": "E2E Slowdown"
    }
    metric_label = metric_names.get(metric, metric)
    
    ax.set_xlabel("Autoscaling Policy", fontsize=11)
    ax.set_ylabel(f"{aggregate.capitalize()} {metric_label} (P{int(quantile * 100)})", fontsize=11)
    ax.set_xticklabels(agg_df["policy"], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {save_path}")
    plt.close()


def main():
    """
    主函数：生成自动扩缩容策略对比图
    """
    # ==================== 配置参数 ====================
    
    # 扩缩容策略列表
    policies = [
        "heteroscale",      # HeteroScale (TPS + Latency)
        "hpa_gpu",          # HPA-GPU (GPU Utilization)
        "independent_tps",  # Independent TPS
        "pure_latency",     # Pure Latency
        
        # "periodic",       # Periodic (可选)
        "no_autoscaling"  # No Autoscaling (可选)
    ]
    
    # 流量追踪列表（根据你的实验配置修改）
    # 示例：rr_conv_10, rr_conv_20, ..., rr_conv_100
    trace_type = "conv"  # 或 "code"
    trace_rates = range(10, 110, 10)  # 10, 20, 30, ..., 100
    traces = [f"rr_{trace_type}_{rate}" for rate in trace_rates]
    
    # 其他配置
    seed = 0
    start_state = "splitwise_25_15"
    cluster = "0_40"
    scheduler = "mixed_pool"
    model = "bloom-176b"
    
    # ==================== 加载数据 ====================
    
    print("正在加载数据...")
    results_df, request_dfs = get_autoscaling_data(
        policies=policies,
        traces=traces,
        seed=seed,
        start_state=start_state,
        cluster=cluster,
        scheduler=scheduler,
        model=model
    )
    
    if results_df.empty:
        print("错误: 没有找到任何数据！请检查路径配置。")
        return
    
    print(f"\n✓ 成功加载 {len(results_df)} 条数据记录")
    print(f"策略: {results_df['policy'].unique()}")
    print(f"流量: {results_df['trace'].unique()}")
    
    # ==================== 生成图表 ====================
    
    # 1. TTFT Slowdown 对比（不同分位数）
    print("\n生成 TTFT Slowdown 对比图...")
    plot_policy_comparison(
        results_df,
        traces,
        metric="ttft_slowdown",
        quantiles=[0.5, 0.9, 0.99],
        title="TTFT Slowdown: Policy Comparison",
        save_path=f"{plots_dir}ttft_slowdown_comparison.png"
    )
    
    # 2. TBT Slowdown 对比（不同分位数）
    print("生成 TBT Slowdown 对比图...")
    plot_policy_comparison(
        results_df,
        traces,
        metric="tbt_slowdown",
        quantiles=[0.5, 0.9, 0.99],
        title="TBT Slowdown: Policy Comparison",
        save_path=f"{plots_dir}tbt_slowdown_comparison.png"
    )
    
    # 3. E2E Slowdown 对比（不同分位数）
    print("生成 E2E Slowdown 对比图...")
    plot_policy_comparison(
        results_df,
        traces,
        metric="e2e_slowdown",
        quantiles=[0.5, 0.9, 0.99],
        title="E2E Slowdown: Policy Comparison",
        save_path=f"{plots_dir}e2e_slowdown_comparison.png"
    )
    
    # 4. 多指标网格对比（P90）
    print("生成多指标网格对比图 (P90)...")
    plot_metrics_grid(
        results_df,
        traces,
        metrics=["ttft_slowdown", "tbt_slowdown", "e2e_slowdown"],
        quantile=0.9,
        title="Performance Metrics Comparison (P90)",
        save_path=f"{plots_dir}metrics_grid_p90.png"
    )
    
    # 5. 柱状图对比（平均值，P90）
    print("生成柱状对比图 (P90 平均值)...")
    plot_bar_comparison(
        results_df,
        metric="tbt_slowdown",
        quantile=0.9,
        aggregate="mean",
        title="TBT Slowdown (P90): Average Across All Traces",
        save_path=f"{plots_dir}bar_comparison_tbt_p90.png"
    )
    
    # 6. 保存数据摘要
    print("\n保存数据摘要...")
    summary_path = f"{plots_dir}results_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"✓ 数据摘要已保存: {summary_path}")
    
    print("\n" + "="*60)
    print("所有图表生成完成！")
    print(f"图表保存在: {plots_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

