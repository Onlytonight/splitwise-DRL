#!/usr/bin/env python3
"""
计算纯token实例和prompt实例的最大 tokens per second (TPS)
基于 splitwise.yaml 配置和性能模型
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from performance_model import DatabasePerformanceModel
import numpy as np

# 初始化性能模型
perf_model = DatabasePerformanceModel('data/perf_model.csv')

# 配置参数（来自 splitwise.yaml）
config = {
    'model': 'llama2-70b',  # 从数据库中可以看到使用的模型
    'max_batch_size': 512,
    'max_batch_tokens': 2048*20,
    'tensor_parallelism': 8,
    'prompt_hardware': 'h100-80gb',  # dgx-h100
    'token_hardware': 'a100-80gb',   # dgx-a100
    'prompt_size_range': [64, 128, 256, 512, 1024, 2048],  # 常见的 prompt_size
    'token_size': 128,  # 用于查询的默认 token_size
}

print("=" * 80)
print("根据 splitwise.yaml 配置和性能模型计算最大 TPS")
print("=" * 80)
print(f"\n配置参数:")
print(f"  - model: {config['model']}")
print(f"  - max_batch_size: {config['max_batch_size']}")
print(f"  - max_batch_tokens: {config['max_batch_tokens']}")
print(f"  - tensor_parallelism: {config['tensor_parallelism']}")
print(f"  - prompt 实例硬件: {config['prompt_hardware']} (dgx-h100)")
print(f"  - token 实例硬件: {config['token_hardware']} (dgx-a100)")

# ============================================================================
# 1. 计算纯 Token 实例的最大 TPS (Decode-only, 使用 a100-80gb)
# ============================================================================
print("\n" + "=" * 80)
print("1. 纯 Token 实例 (Decode-only) 最大 TPS")
print("=" * 80)

print(f"\n硬件: {config['token_hardware']}")
print(f"模型: {config['model']}")
print(f"Tensor Parallelism: {config['tensor_parallelism']}")

# 测试不同的 batch_size
batch_sizes_to_test = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# 过滤掉超过限制的 batch_size
batch_sizes_to_test = [bs for bs in batch_sizes_to_test if bs <= config['max_batch_size']]

print(f"\n测试的 batch sizes: {batch_sizes_to_test}")
print("\n对于纯 token 实例，每个迭代处理 batch_size 个 tokens（每个任务生成1个token）")
print("TPS = batch_size / token_time_per_iteration\n")

token_results = []
print("Batch Size | Token Time (ms) | Token Time (s) | TPS (tokens/s) | Batch Tokens")
print("-" * 80)

for batch_size in batch_sizes_to_test:
    # 对于纯 decode，batch_tokens = batch_size（每个任务1个token）
    batch_tokens = batch_size
    
    # 检查是否超过 max_batch_tokens
    if batch_tokens > config['max_batch_tokens']:
        continue
    
    try:
        # 使用性能模型查询 token 时间
        token_time_sec = perf_model.get_token_time(
            model=config['model'],
            hardware=config['token_hardware'],
            tensor_parallel=config['tensor_parallelism'],
            prompt_size=512,  # prompt_size 对 token_time 影响较小，使用典型值
            batch_size=batch_size,
            token_size=config['token_size'],
            batch_tokens=batch_tokens
        )
        
        # 计算 TPS：每次迭代处理 batch_size 个 tokens
        tps = batch_size / token_time_sec
        
        token_results.append({
            'batch_size': batch_size,
            'batch_tokens': batch_tokens,
            'token_time_sec': token_time_sec,
            'token_time_ms': token_time_sec * 1000,
            'tps': tps
        })
        
        print(f"{batch_size:10d} | {token_time_sec*1000:15.2f} | {token_time_sec:14.6f} | {tps:14.2f} | {batch_tokens:12d}")
    
    except Exception as e:
        print(f"{batch_size:10d} | Error: {e}")

if token_results:
    # 找到最大 TPS
    max_tps_result = max(token_results, key=lambda x: x['tps'])
    
    print("\n" + "=" * 80)
    print(f"✓ 纯 Token 实例最大 TPS: {max_tps_result['tps']:.2f} tokens/s")
    print(f"  - 最优 batch size: {max_tps_result['batch_size']}")
    print(f"  - 每次迭代时间: {max_tps_result['token_time_ms']:.2f} ms ({max_tps_result['token_time_sec']:.6f} s)")
    print(f"  - Batch tokens: {max_tps_result['batch_tokens']}")
    print("=" * 80)
    
    # 检查是否与用户提到的时间一致
    user_mentioned_time_ms = 34  # 用户提到的 0.034ms 应该是 34ms
    if abs(max_tps_result['token_time_ms'] - user_mentioned_time_ms) < 10:
        print(f"\n✓ 与用户提到的 ~{user_mentioned_time_ms}ms 相符")
    else:
        print(f"\n注意: 数据库中的值 ({max_tps_result['token_time_ms']:.2f}ms) 与用户提到的 {user_mentioned_time_ms}ms 有差异")
        # 找到最接近用户提到时间的配置
        closest_result = min(token_results, key=lambda x: abs(x['token_time_ms'] - user_mentioned_time_ms))
        if closest_result != max_tps_result:
            print(f"\n最接近 {user_mentioned_time_ms}ms 的配置:")
            print(f"  - Batch size: {closest_result['batch_size']}")
            print(f"  - Token time: {closest_result['token_time_ms']:.2f} ms")
            print(f"  - TPS: {closest_result['tps']:.2f} tokens/s")

# ============================================================================
# 2. 计算 Prompt 实例的最大 TPS (Prefill, 使用 h100-80gb)
# ============================================================================
print("\n" + "=" * 80)
print("2. Prompt 实例 (Prefill) 最大 TPS")
print("=" * 80)

print(f"\n硬件: {config['prompt_hardware']}")
print(f"模型: {config['model']}")
print(f"Tensor Parallelism: {config['tensor_parallelism']}")

print(f"\n注意: Prompt 实例的 TPS 取决于输入大小 (prompt_size)")
print(f"测试的 prompt sizes: {config['prompt_size_range']}")
print("\n对于 prompt 实例，每次处理 batch_size × prompt_size 个 tokens")
print("TPS = (batch_size × prompt_size) / prompt_time\n")

prompt_results = []

print("Batch Size | Prompt Size | Total Tokens | Prompt Time (ms) | Prompt Time (s) | TPS (tokens/s)")
print("-" * 105)

# 测试不同的 prompt_size 和 batch_size 组合
for prompt_size in config['prompt_size_range']:
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        # 检查限制
        if batch_size > config['max_batch_size']:
            continue
        
        total_tokens = batch_size * prompt_size
        
        # 检查是否超过 max_batch_tokens
        if total_tokens > config['max_batch_tokens']:
            continue
        
        try:
            # 使用性能模型查询 prompt 时间
            prompt_time_sec = perf_model.get_prompt_time(
                model=config['model'],
                hardware=config['prompt_hardware'],
                tensor_parallel=config['tensor_parallelism'],
                prompt_size=prompt_size,
                batch_size=batch_size,
                token_size=config['token_size'],
                batch_tokens=total_tokens
            )
            
            # 计算 TPS
            tps = total_tokens / prompt_time_sec
            
            prompt_results.append({
                'batch_size': batch_size,
                'prompt_size': prompt_size,
                'total_tokens': total_tokens,
                'prompt_time_sec': prompt_time_sec,
                'prompt_time_ms': prompt_time_sec * 1000,
                'tps': tps
            })
            
            # 只显示 TPS 较高的配置（Top 20）
            
        except Exception as e:
            pass  # 跳过查询失败的配置

# 按 TPS 降序排列，显示前 20 个
# prompt_results_sorted = sorted(prompt_results, key=lambda x: x['tps'], reverse=True)[:200]

for result in prompt_results:
    print(f"{result['batch_size']:10d} | {result['prompt_size']:11d} | {result['total_tokens']:12d} | "
          f"{result['prompt_time_ms']:16.2f} | {result['prompt_time_sec']:15.6f} | {result['tps']:14.2f}")

if prompt_results:
    max_prompt_tps_result = max(prompt_results, key=lambda x: x['tps'])
    
    print("\n" + "=" * 105)
    print(f"✓ Prompt 实例最大 TPS: {max_prompt_tps_result['tps']:.2f} tokens/s")
    print(f"  - 最优配置: batch_size={max_prompt_tps_result['batch_size']}, prompt_size={max_prompt_tps_result['prompt_size']}")
    print(f"  - 总 tokens: {max_prompt_tps_result['total_tokens']}")
    print(f"  - Prompt 处理时间: {max_prompt_tps_result['prompt_time_ms']:.2f} ms ({max_prompt_tps_result['prompt_time_sec']:.6f} s)")
    print("=" * 105)
    
    # 显示不同 prompt_size 的最优配置
    print("\n不同 prompt_size 下的最优 TPS:")
    print("-" * 80)
    for prompt_size in config['prompt_size_range']:
        configs_for_size = [r for r in prompt_results if r['prompt_size'] == prompt_size]
        if configs_for_size:
            best_for_size = max(configs_for_size, key=lambda x: x['tps'])
            print(f"  prompt_size={prompt_size:4d}: TPS={best_for_size['tps']:8.2f} tokens/s "
                  f"(batch_size={best_for_size['batch_size']}, "
                  f"time={best_for_size['prompt_time_ms']:.2f}ms)")

# ============================================================================
# 3. 总结和 HeteroScale 配置建议
# ============================================================================
print("\n" + "=" * 80)
print("3. 总结和 HeteroScale 配置建议")
print("=" * 80)

if token_results:
    max_token_tps = max_tps_result['tps']
    print(f"\n✓ 单个 Token 实例的最大 TPS: {max_token_tps:.2f} tokens/s")
    print(f"  (在 batch_size={max_tps_result['batch_size']} 时达到)")
    
    print(f"\n推荐 HeteroScale 配置 (configs/simulator/baseline_heteroscale.yaml):")
    print(f"\npolicy_config:")
    print(f"  target_decode_tps_per_instance: {max_token_tps:.0f}  # 每个 token 实例的目标 TPS")
    
    # 保守估计（考虑实际负载可能达不到理论最大值）
    conservative_tps = max_token_tps * 0.8  # 80% 利用率
    moderate_tps = max_token_tps * 0.9  # 90% 利用率
    
    print(f"\n或使用保守估计:")
    print(f"  target_decode_tps_per_instance: {conservative_tps:.0f}  # 80% 利用率（更稳定）")
    print(f"  target_decode_tps_per_instance: {moderate_tps:.0f}  # 90% 利用率（平衡）")

if prompt_results:
    max_prompt_tps = max_prompt_tps_result['tps']
    print(f"\n✓ 单个 Prompt 实例的最大 TPS: {max_prompt_tps:.2f} tokens/s")
    print(f"  (在最优配置下: batch_size={max_prompt_tps_result['batch_size']}, "
          f"prompt_size={max_prompt_tps_result['prompt_size']})")
    
    # 计算不同 prompt_size 的平均 TPS
    avg_tps_by_size = {}
    for prompt_size in config['prompt_size_range']:
        configs_for_size = [r for r in prompt_results if r['prompt_size'] == prompt_size]
        if configs_for_size:
            avg_tps = np.mean([r['tps'] for r in configs_for_size])
            avg_tps_by_size[prompt_size] = avg_tps
    
    print(f"\n注意:")
    print(f"  - Prompt TPS 高度依赖于实际请求的 prompt_size 分布")
    print(f"  - 典型的 prompt_size 范围: 64-512 tokens")
    print(f"  - 在实际工作负载中，prompt_size 越小，延迟越低，但 TPS 也越低")

print("\n" + "=" * 80)
print("计算完成！")
print("=" * 80)
