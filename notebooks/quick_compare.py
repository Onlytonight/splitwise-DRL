"""
快速对比不同自动扩缩容策略 - 简化版
使用方法：python quick_compare.py
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置样式
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10})

# ==================== 配置区域 ====================
# 请根据你的实验配置修改以下参数

# 1. 基础路径
RESULTS_DIR = "../results"
PLOTS_DIR = "../plots/autoscaling_quick/"
os.makedirs(PLOTS_DIR, exist_ok=True)

# 2. 实验配置
SEED = 0
START_STATE = "splitwise_25_15"
CLUSTER = "0_40"
SCHEDULER = "mixed_pool"
MODEL = "bloom-176b"

# 3. 扩缩容策略（你想对比的策略）
POLICIES = [
    "heteroscale",
    "hpa_gpu",
    "independent_tps",
    "pure_latency",
]

# 4. 流量追踪（根据你的实验选择）
TRACE_TYPE = "conv"  # 或 "code"
TRACE_RATES = [10, 20, 30, 40, 50]  # 请求速率列表

# ==================== 主程序 ====================

def load_summary_csv(policy, trace):
    """加载单个策略和流量的 summary.csv"""
    path = f"{RESULTS_DIR}/{SEED}/{START_STATE}/{trace}/{CLUSTER}/{MODEL}/{SCHEDULER}/{policy}/summary.csv"
    
    if not os.path.exists(path):
        return None
    
    try:
        df = pd.read_csv(path)
        df['policy'] = policy
        df['trace'] = trace
        df['rate'] = int(trace.split('_')[-1])
        return df
    except Exception as e:
        print(f"警告: 无法读取 {path}: {e}")
        return None


def collect_all_data():
    """收集所有数据"""
    all_data = []
    
    traces = [f"rr_{TRACE_TYPE}_{rate}" for rate in TRACE_RATES]
    
    print("正在收集数据...")
    for policy in POLICIES:
        for trace in traces:
            df = load_summary_csv(policy, trace)
            if df is not None:
                all_data.append(df)
                print(f"✓ {policy} - {trace}")
            else:
                print(f"✗ {policy} - {trace} (未找到)")
    
    if not all_data:
        print("\n错误: 没有找到任何数据！")
        print(f"请检查路径: {RESULTS_DIR}/{SEED}/{START_STATE}/")
        sys.exit(1)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✓ 成功收集 {len(combined_df)} 条记录")
    return combined_df


def plot_comparison(df, metric, ylabel, title, filename):
    """绘制对比图"""
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    quantiles = [50, 90, 99]
    
    for i, q in enumerate(quantiles):
        col = f"{metric}_p{q}"
        
        if col not in df.columns:
            print(f"警告: 列 {col} 不存在，跳过")
            continue
        
        # 绘制折线图
        for policy in POLICIES:
            policy_data = df[df['policy'] == policy].sort_values('rate')
            if len(policy_data) > 0:
                axs[i].plot(policy_data['rate'], policy_data[col], 
                           marker='o', label=policy, linewidth=2, markersize=6)
        
        axs[i].set_title(f"P{q}", fontweight='bold')
        axs[i].set_xlabel("Request Rate (req/s)")
        axs[i].set_ylabel(ylabel if i == 0 else "")
        axs[i].grid(True, alpha=0.3)
        axs[i].legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}{filename}", dpi=300, bbox_inches='tight')
    print(f"✓ 保存: {filename}")
    plt.close()


def plot_summary_table(df):
    """生成汇总表格"""
    # 计算每个策略的平均指标
    summary_data = []
    
    for policy in POLICIES:
        policy_df = df[df['policy'] == policy]
        
        if len(policy_df) == 0:
            continue
        
        row = {
            'Policy': policy,
            'TTFT P90': policy_df['ttft_times_p90'].mean() if 'ttft_times_p90' in policy_df else 0,
            'TBT P90': policy_df['tbt_times_p90'].mean() if 'tbt_times_p90' in policy_df else 0,
            'E2E P90': policy_df['response_times_p90'].mean() if 'response_times_p90' in policy_df else 0,
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存为 CSV
    csv_path = f"{PLOTS_DIR}summary_table.csv"
    summary_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ 保存汇总表: {csv_path}")
    
    # 打印到控制台
    print("\n" + "="*60)
    print("汇总表 (平均值)")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)


def main():
    """主函数"""
    print("="*60)
    print("自动扩缩容策略快速对比工具")
    print("="*60)
    print(f"\n配置:")
    print(f"  - 策略: {', '.join(POLICIES)}")
    print(f"  - 流量: {TRACE_TYPE}_{TRACE_RATES}")
    print(f"  - 调度器: {SCHEDULER}")
    print(f"  - 模型: {MODEL}")
    print()
    
    # 1. 收集数据
    df = collect_all_data()
    
    # 2. 生成图表
    print("\n生成对比图表...")
    
    # TTFT
    if 'ttft_times_p50' in df.columns:
        plot_comparison(df, 'ttft_times', 'TTFT (seconds)', 
                       'Time to First Token Comparison', 'ttft_comparison.png')
    
    # TBT
    if 'tbt_times_p50' in df.columns:
        plot_comparison(df, 'tbt_times', 'TBT (seconds)', 
                       'Time Between Tokens Comparison', 'tbt_comparison.png')
    
    # E2E
    if 'response_times_p50' in df.columns:
        plot_comparison(df, 'response_times', 'E2E Time (seconds)', 
                       'End-to-End Response Time Comparison', 'e2e_comparison.png')
    
    # 3. 生成汇总表
    plot_summary_table(df)
    
    print(f"\n✓ 完成！图表保存在: {PLOTS_DIR}")


if __name__ == "__main__":
    main()

