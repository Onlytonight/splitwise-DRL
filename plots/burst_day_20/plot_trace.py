"""
读取 traces/burst/day_20.csv 并绘制 prompt_size, token_size, token_size+prompt_size 随时间的折线图
分成三个子图，每个图带误差棒
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_trace_data(csv_file='../../traces/burst/day_13.csv', output_file='day_13_trace_plot.png',
                    time_window=0.5):
    """
    读取CSV文件并绘制 prompt_size, token_size, token_size+prompt_size 随时间的折线图
    分成三个子图，每个图带误差棒
    
    参数:
    csv_file: CSV文件路径
    output_file: 输出图像路径
    time_window: 时间窗口大小（秒），用于计算误差棒
    """
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 读取CSV文件
    print(f"正在读取文件: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 检查必要的列是否存在
    required_columns = ['arrival_timestamp', 'prompt_size', 'token_size']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要的列: {missing_columns}")
    
    # 提取数据
    time = df['arrival_timestamp'].values
    prompt_size = df['prompt_size'].values
    token_size = df['token_size'].values
    total_size = token_size + prompt_size
    
    # 按时间窗口聚合数据以计算误差棒
    def aggregate_with_error(data, time, window_size):
        """按时间窗口聚合数据，返回均值、标准差和时间点"""
        # 创建时间窗口
        time_bins = np.arange(time.min(), time.max() + window_size, window_size)
        bin_indices = np.digitize(time, time_bins) - 1
        
        # 计算每个窗口的统计量
        bin_means = []
        bin_stds = []
        bin_centers = []
        
        for i in range(len(time_bins) - 1):
            mask = bin_indices == i
            if np.any(mask):
                bin_means.append(np.mean(data[mask]))
                bin_stds.append(np.std(data[mask]))
                bin_centers.append((time_bins[i] + time_bins[i+1]) / 2)
        
        return np.array(bin_centers), np.array(bin_means), np.array(bin_stds)
    
    # 聚合数据
    time_prompt, mean_prompt, std_prompt = aggregate_with_error(prompt_size, time, time_window)
    time_token, mean_token, std_token = aggregate_with_error(token_size, time, time_window)
    time_total, mean_total, std_total = aggregate_with_error(total_size, time, time_window)
    
    # 创建三个子图
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 子图1: prompt_size
    axes[0].errorbar(time_prompt, mean_prompt, yerr=std_prompt, 
                     label='prompt_size', linewidth=1.5, alpha=0.8, color='blue',
                     capsize=2, capthick=1, elinewidth=1)
    axes[0].set_xlabel('时间 (arrival_timestamp)', fontsize=11)
    axes[0].set_ylabel('prompt_size', fontsize=11)
    axes[0].set_title('prompt_size 随时间的变化（带误差棒）', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle=':')
    
    # 子图2: token_size
    axes[1].errorbar(time_token, mean_token, yerr=std_token,
                     label='token_size', linewidth=1.5, alpha=0.8, color='green',
                     capsize=2, capthick=1, elinewidth=1)
    axes[1].set_xlabel('时间 (arrival_timestamp)', fontsize=11)
    axes[1].set_ylabel('token_size', fontsize=11)
    axes[1].set_title('token_size 随时间的变化（带误差棒）', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle=':')
    
    # 子图3: token_size + prompt_size
    axes[2].errorbar(time_total, mean_total, yerr=std_total,
                     label='token_size + prompt_size', linewidth=1.5, alpha=0.8, color='red',
                     capsize=2, capthick=1, elinewidth=1)
    axes[2].set_xlabel('时间 (arrival_timestamp)', fontsize=11)
    axes[2].set_ylabel('token_size + prompt_size', fontsize=11)
    axes[2].set_title('token_size + prompt_size 随时间的变化（带误差棒）', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, linestyle=':')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_file}")
    
    # 关闭图表以释放内存
    plt.close()
    
    # 如果需要显示图表，取消下面的注释
    # plt.show()

if __name__ == "__main__":
    plot_trace_data()

