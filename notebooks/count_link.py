import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def count_dummy_links(rr_conv_value, pool_type):
    """
    统计指定 rr_conv 值目录下指定 pool_type 目录中 request_nodes.csv 中 DummyLink 的数量
    """
    base_path = f"/home/xfusion/conda/splitwise-DRL/results/0/splitwise_25_15/rr_conv_{rr_conv_value}/0_40/bloom-176b/{pool_type}"
    file_path = os.path.join(base_path, "request_nodes.csv")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return 0
    
    # 读取文件并统计 DummyLink 数量
    try:
        df = pd.read_csv(file_path)
        dummy_link_count = len(df[df['runner'] == 'DummyLink'])
        return dummy_link_count
    except Exception as e:
        print(f"读取文件时出错 {file_path}: {e}")
        return 0

def main():
    # 定义要统计的 rr_conv 值范围 (30 到 250，步长为 10)
    rr_conv_values = list(range(30, 260, 10))  # 30, 40, 50, ..., 250
    unified_pool_counts = []
    mixed_pool_counts = []
    
    # 统计每个 rr_conv 值对应的 DummyLink 数量
    for rr_conv in rr_conv_values:
        # 统计 Unified_pool 中的 DummyLink 数量
        unified_count = count_dummy_links(rr_conv, "Unified_pool")
        unified_pool_counts.append(unified_count)
        
        # 统计 mixed_pool 中的 DummyLink 数量
        mixed_count = count_dummy_links(rr_conv, "mixed_pool")
        mixed_pool_counts.append(mixed_count)
        
        print(f"rr_conv_{rr_conv}: Unified_pool={unified_count}, mixed_pool={mixed_count}")
    
    # 绘制图表
    plt.figure(figsize=(12, 6))
    plt.plot(rr_conv_values, unified_pool_counts, marker='o', linestyle='-', linewidth=2, markersize=8, label='Unified_pool')
    plt.plot(rr_conv_values, mixed_pool_counts, marker='s', linestyle='--', linewidth=2, markersize=8, label='mixed_pool')
    plt.xlabel('rr_conv 值')
    plt.ylabel('DummyLink 数量')
    plt.title('不同 rr_conv 值下 Unified_pool 和 mixed_pool 的 DummyLink 数量统计')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('/home/xfusion/conda/splitwise-DRL/notebooks/dummy_link_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印汇总信息
    print("\n统计汇总:")
    print(f"总共处理了 {len(rr_conv_values)} 个目录")
    print(f"Unified_pool 中 DummyLink 总数: {sum(unified_pool_counts)}")
    print(f"mixed_pool 中 DummyLink 总数: {sum(mixed_pool_counts)}")
    print(f"Unified_pool 平均每个目录的 DummyLink 数量: {np.mean(unified_pool_counts):.2f}")
    print(f"mixed_pool 平均每个目录的 DummyLink 数量: {np.mean(mixed_pool_counts):.2f}")

if __name__ == "__main__":
    main()