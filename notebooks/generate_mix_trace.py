import numpy as np
import pandas as pd
import scipy.stats as stats
from collections import namedtuple
import os

# 保持原有定义不变
Distributions = namedtuple('Distributions', ['application_id',
                                             'request_type',
                                             'arrival_process',
                                             'batch_size',
                                             'prompt_size',
                                             'token_size'])
Distribution = namedtuple('Distribution', ['name', 'params'])


# 辅助函数：保持原有 generate_samples 逻辑，但为了混合，我们需要能从指定文件采样
def load_trace_data(filename, column, size):
    df = pd.read_csv(filename)
    # 如果文件行数不够，允许 replace=True
    return df[column].sample(size, replace=True).values


def generate_samples(distribution, params, size):
    """
    Generate random samples from the given distribution.
    Refactored to handle direct passing of trace data if needed, 
    but keeping original logic for compatibility.
    """
    if distribution == "constant":
        return np.ones(size) * params["value"]
    elif distribution == "exponential":
        return stats.expon(**params).rvs(size=size)
    elif distribution == "trace":
        return load_trace_data(params["filename"], params["column"], size)
    # ... (其他原有分布保持不变)
    else:
        # Fallback for standard scipy distributions
        if hasattr(stats, distribution):
            dist_func = getattr(stats, distribution)
            return dist_func(**params).rvs(size=size)
        raise ValueError(f"Invalid distribution: {distribution}")


def generate_mixed_trace_from_distributions(
        max_requests,
        end_time,
        total_request_rate,
        mix_ratio,  # 0.0 - 1.0, 表示 Code 负载的占比 (e.g., 0.3 means 30% Code)
        code_dist_file,
        chat_dist_file
):
    """
    Generate a scientifically mixed trace combining Code and Chat workloads.

    Theory:
    Superposition of Poisson Processes. We generate a unified arrival stream 
    with lambda_total, and use a Bernoulli trial (mix_ratio) to assign 
    workload types. This preserves the burstiness characteristics relative 
    to the aggregate load.
    """

    # 1. 生成统一的到达时间流 (Global Arrival Process)
    # 混合后的流依然服从指数分布间隔
    arrival_interarrival = stats.expon(scale=1.0 / total_request_rate).rvs(size=max_requests)
    arrival_timestamps = np.cumsum(arrival_interarrival)

    # 如果有 end_time 限制，提前截断以减少计算
    if end_time is not None:
        valid_indices = arrival_timestamps < end_time
        arrival_timestamps = arrival_timestamps[valid_indices]
        real_count = len(arrival_timestamps)
    else:
        real_count = max_requests

    # 2. 确定每个请求的类型 (Bernoulli Trial)
    # 生成一个 mask，True 代表 Code，False 代表 Chat
    # random_vals < mix_ratio  --> Code
    is_code_mask = np.random.rand(real_count) < mix_ratio
    num_code = np.sum(is_code_mask)
    num_chat = real_count - num_code

    print(f"Generating Mixed Trace: Total={real_count}, Code={num_code} ({num_code / real_count:.2%}), Chat={num_chat}")

    # 3. 初始化结果数组
    request_ids = np.arange(real_count)# 0 for Chat, 1 for Code (可自定义)
    application_ids = np.zeros(real_count, dtype=int)
    request_types = np.full(real_count, 2, dtype=int)  # 2 for LLM inference
    batch_sizes = np.ones(real_count, dtype=int)

    # 预分配 Prompt 和 Token 数组
    prompt_sizes = np.zeros(real_count, dtype=int)
    token_sizes = np.zeros(real_count, dtype=int)

    # 4. 分别采样并填充 (Vectorized Sampling)
    # 这种方法比循环快得多，且符合 Scientific Computing 习惯

    # --- Fill Code Requests ---
    if num_code > 0:
        # application_ids[is_code_mask] = 1  # 标记 Code 为 App ID 1
        prompt_sizes[is_code_mask] = load_trace_data(code_dist_file, "ContextTokens", num_code)
        token_sizes[is_code_mask] = load_trace_data(code_dist_file, "GeneratedTokens", num_code)

    # --- Fill Chat Requests ---
    if num_chat > 0:
        # application_ids[~is_code_mask] = 0  # 标记 Chat 为 App ID 0
        prompt_sizes[~is_code_mask] = load_trace_data(chat_dist_file, "ContextTokens", num_chat)
        token_sizes[~is_code_mask] = load_trace_data(chat_dist_file, "GeneratedTokens", num_chat)

    # 5. 组装 DataFrame
    trace_df = pd.DataFrame({
        "request_id": request_ids,
        "request_type": request_types,
        "application_id": application_ids,  # 关键：调度器据此区分 Code/Chat
        "arrival_timestamp": arrival_timestamps,
        "batch_size": batch_sizes,
        "prompt_size": prompt_sizes,
        "token_size": token_sizes,
        # 额外添加一个 human-readable 的标签，方便 debug 和画图
        "workload_type": [0 if x else 1 for x in is_code_mask] #1chat 0code
    })

    return trace_df


def generate_mixed_traces_batch(
        max_requests,
        end_time,
        request_rates,
        mix_ratios,  # list of ratios, e.g., [0.1, 0.5, 0.9]
        code_file,
        chat_file,
        trace_filename_template="../traces/mixed_r{}_ratio{}.csv"
):
    """
    Batch generator for sensitivity analysis (sweep over Rate and Ratio).
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]):
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])

    for rate in request_rates:
        for ratio in mix_ratios:
            print(f"Generating -> Rate: {rate}, Code Ratio: {ratio}")
            df = generate_mixed_trace_from_distributions(
                max_requests=max_requests,
                end_time=end_time,
                total_request_rate=rate,
                mix_ratio=ratio,
                code_dist_file=code_file,
                chat_dist_file=chat_file
            )

            # 文件名包含 rate 和 ratio，方便实验脚本解析
            # ratio 乘以 100 变成整数方便命名，例如 0.3 -> 30
            filename = trace_filename_template.format(rate, int(ratio * 100))
            df.to_csv(filename, index=False)


# --- 使用示例 ---

if __name__ == "__main__":
    # 假设你已经下载好了 Azure 的 trace 数据
    code_file = "../data/code_distributions.csv"
    chat_file = "../data/conv_distributions.csv"

    # 如果文件不存在，先下载 (复用你原有的逻辑)
    # ... (download logic) ...

    # 设定实验参数
    # 场景 1: 固定 QPS=10，扫描 Code 比例 (0% -> 100%)
    # 这对于证明你的系统在混合负载变化时的鲁棒性至关重要
    # ratios_to_test = [0.0, 0.2, 0.5, 0.8, 1.0]
    # fixed_rate = [50]
    #
    # generate_mixed_traces_batch(
    #     max_requests=100000,
    #     end_time=600,  # 10分钟
    #     request_rates=fixed_rate,
    #     mix_ratios=ratios_to_test,
    #     code_file=code_file,
    #     chat_file=chat_file,
    #     trace_filename_template="../traces/mixed_qps{}_code{}.csv"
    # )

    # 场景 2: 固定 Code 比例 30% (典型 Copilot 场景)，扫描 QPS 压力
    # 用于测试 Throughput 和 Latency 曲线
    rates_to_test = list(range(30, 151, 10))
    fixed_ratio = [0.3]

    generate_mixed_traces_batch(
        max_requests=100000,
        end_time=600,
        request_rates=rates_to_test,
        mix_ratios=fixed_ratio,
        code_file=code_file,
        chat_file=chat_file,
        trace_filename_template="../traces/mixed_qps_{}_code{}.csv"
    )