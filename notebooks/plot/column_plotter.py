import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from scipy.ndimage import gaussian_filter1d

# 支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_second_column(csv_file, output_dir='./plots/'):
    """
    读取CSV文件的第二列并绘制折线图和平滑曲线图，跳过第一行
    
    参数:
    csv_file: CSV文件路径
    output_dir: 输出图像的目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 读取CSV文件，跳过第一行
        df = pd.read_csv(csv_file, skiprows=1)
        
        # 检查是否有足够的列
        if df.shape[1] < 2:
            raise ValueError(f"文件 {csv_file} 列数不足，至少需要2列")
            
        # 提取第二列数据
        second_column = df.iloc[:, 1]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制原始折线图
        ax1.plot(second_column, linewidth=1)
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')
        ax1.set_title('Column 2 Data - Original Line Plot')
        ax1.grid(True, alpha=0.3)
        
        # 绘制平滑曲线图
        smoothed_data = gaussian_filter1d(second_column, sigma=30)
        ax2.plot(smoothed_data, linewidth=2, color='red')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.set_title('Column 2 Data - Smoothed Curve')
        ax2.grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        filename = os.path.basename(csv_file)
        output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_column2_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"已保存图表到 {output_path}")
        
    except Exception as e:
        print(f"处理文件 {csv_file} 时出错: {e}")

def plot_multiple_files(csv_files, output_dir='./plots/'):
    """
    读取多个CSV文件的第二列并绘制在同一张图上
    
    参数:
    csv_files: CSV文件路径列表
    output_dir: 输出图像的目录路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    for csv_file in csv_files:
        try:
            # 读取CSV文件，跳过第一行
            df = pd.read_csv(csv_file, skiprows=1)
            
            # 检查是否有足够的列
            if df.shape[1] < 2:
                print(f"警告: 文件 {csv_file} 列数不足，跳过")
                continue
                
            # 提取第二列数据
            second_column = df.iloc[:, 1]
            filename = os.path.basename(csv_file)
            
            # 绘制原始折线图
            ax1.plot(second_column, label=filename, linewidth=1, alpha=0.7)
            
            # 绘制平滑曲线图
            smoothed_data = gaussian_filter1d(second_column, sigma=30)
            ax2.plot(smoothed_data, label=filename, linewidth=2, alpha=0.7)
            
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")
    
    # 设置图表属性
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    ax1.set_title('Column 2 Data - Original Line Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    ax2.set_title('Column 2 Data - Smoothed Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.join(output_dir, "combined_column2_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存合并图表到 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='读取CSV文件的第二列并绘制折线图和平滑曲线图')
    parser.add_argument('--csv_file', type=str, help='单个CSV文件路径')
    parser.add_argument('--csv_files', type=str, nargs='+', help='多个CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='./plots/',
                        help='输出图像的目录路径')
    
    args = parser.parse_args()
    
    if args.csv_file:
        plot_second_column(args.csv_file, args.output_dir)
    elif args.csv_files:
        plot_multiple_files(args.csv_files, args.output_dir)
    else:
        print("请提供 --csv_file 或 --csv_files 参数")