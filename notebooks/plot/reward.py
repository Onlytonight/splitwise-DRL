import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from scipy.ndimage import gaussian_filter1d

def plot_reward_curve(csv_dir, x_dir='x'):
    """
    读取指定目录下的CSV文件，绘制奖励曲线图
    
    参数:
    csv_dir: 包含CSV文件的目录路径
    x_dir: 输出图像的目录路径
    """
    # 创建输出目录
    os.makedirs(x_dir, exist_ok=True)
    
    # 查找CSV文件
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"在目录 {csv_dir} 中未找到CSV文件")
        return
    
    for csv_file in csv_files:
        file_path = os.path.join(csv_dir, csv_file)
        
        try:
            # 读取CSV文件，跳过第一行
            df = pd.read_csv(file_path, skiprows=1)
            
            # 检查是否有足够的列
            if df.shape[1] < 2:
                print(f"文件 {csv_file} 列数不足，跳过")
                continue
                
            # 提取第二列作为奖励值
            reward_column = df.iloc[:, 1]
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制原始折线图
            ax.plot(reward_column, label='原始奖励', linewidth=1, alpha=0.7)
            
            # 绘制平滑曲线图
            smoothed_rewards = gaussian_filter1d(reward_column, sigma=5)
            ax.plot(smoothed_rewards, label='平滑奖励 (σ=5)', linewidth=2)
            
            # 设置图表属性
            ax.set_xlabel('步骤')
            ax.set_ylabel('奖励值')
            ax.set_title(f'{csv_file} 奖励曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 保存图像
            output_path = os.path.join(x_dir, f"{os.path.splitext(csv_file)[0]}_reward_curve.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已保存 {output_path}")
            
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")

if __name__ == "__main__":
    csv_dir = './results/0/splitwise_10_10/long_rps_code_combined/0_20/bloom-176b/mixed_pool/'
    output_dir = './plots/'
    plot_reward_curve(csv_dir, output_dir)