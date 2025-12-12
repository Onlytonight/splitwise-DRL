import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

# 复用原本的下载逻辑
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)

def download_azure_llm_traces():
    if not os.path.exists("data"):
        os.makedirs("data")
    url_base = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/"
    if not os.path.exists("data/code_distributions.csv"):
        print("Downloading code distributions...")
        download_file(url_base + "AzureLLMInferenceTrace_code.csv", "data/code_distributions.csv")
    if not os.path.exists("data/conv_distributions.csv"):
        print("Downloading conv distributions...")
        download_file(url_base + "AzureLLMInferenceTrace_conv.csv", "data/conv_distributions.csv")

class SyntheticTraceGenerator:
    def __init__(self, code_file, conv_file, seed=42):
        self.code_df = pd.read_csv(code_file)
        self.conv_df = pd.read_csv(conv_file)
        np.random.seed(seed)
        
    def generate_rps_curve(self, duration_sec, base_min=10, base_max=50, num_bursts=120):
        """
        生成每秒的 RPS (Requests Per Second) 曲线
        混合模式：潮汐 + 噪音 + 随机突发
        """
        time_steps = np.arange(duration_sec)
        
        # 1. 潮汐效应 (Diurnal): 使用两个叠加的正弦波，模拟非单调的变化
        # 周期分别为 duration 和 duration/3
        wave1 = np.sin(time_steps * (2 * np.pi / duration_sec)) 
        wave2 = 0.5 * np.sin(time_steps * (2 * np.pi / (duration_sec / 3)))
        trend = (wave1 + wave2) 
        # 归一化到 0~1
        trend = (trend - trend.min()) / (trend.max() - trend.min())
        
        # 映射到 base_min ~ base_max
        rps_base = base_min + trend * (base_max - base_min)
        
        # 2. 随机噪音 (Jitter): +/- 15% 的波动
        noise = np.random.normal(0, 0.15, size=duration_sec)
        rps_noisy = rps_base * (1 + noise)
        rps_noisy = np.maximum(rps_noisy, 5) # 保证最小有 5 RPS
        
        # 3. 突发流量 (Bursts): 必须包含至少 num_bursts 次
        # 随机选择突发开始时间
        burst_starts = np.random.choice(np.arange(0, duration_sec - 60), size=num_bursts, replace=False)
        
        final_rps = rps_noisy.copy()
        
        print(f"Injecting {num_bursts} bursts...")
        for start_t in burst_starts:
            # 持续 15s 到 45s
            duration = np.random.randint(30, 45)
            # 基础RPS平均值的2.0x 到 5.0x
            intensity = np.random.uniform(1.0, 2.0)
            
            # 使用高斯核平滑突发边缘，避免过于突兀导致仿真器崩溃
            x = np.linspace(-3, 3, duration)
            burst_shape = np.exp(-x**2) # 高斯钟形
            
            end_t = min(start_t + duration, duration_sec)
            actual_len = end_t - start_t
            
            # 叠加突发
            peak_val = final_rps[start_t:end_t].mean() * intensity
            final_rps[start_t:end_t] += burst_shape[:actual_len] * peak_val

        return final_rps

    def generate_mix_ratio_curve(self, duration_sec):
        """
        生成混合比例曲线 alpha(t)。
        alpha 代表 Code (Prompt-heavy) 请求的占比。
        1 - alpha 代表 Conv (Token-heavy) 请求的占比。
        """
        time_steps = np.arange(duration_sec)
        
        # 慢速变化，模拟不同时间段用户行为的改变
        # 范围在 0.2 (大部分是聊天) 到 0.8 (大部分是写代码) 之间
        ratio = 0.5 + 0.3 * np.sin(time_steps * (2 * np.pi / (duration_sec / 2)))
        
        # 加上一点随机游走
        noise = np.random.normal(0, 0.05, size=duration_sec)
        ratio = np.clip(ratio + noise, 0.1, 0.9)
        
        return ratio

    def generate_trace(self, duration_sec=14400, output_file="rl_training_trace.csv"):
        """
        主生成函数
        """
        print(f"Generating RL Training Trace for {duration_sec} seconds...")
        
        # 1. 生成控制曲线
        # 14400秒 (4小时) 内 120次突发，平均2分钟一次，密度适中
        rps_curve = self.generate_rps_curve(duration_sec, base_min=10, base_max=50, num_bursts=120)
        ratio_curve = self.generate_mix_ratio_curve(duration_sec)
        
        # 绘制预览图 (可选，方便确认分布)
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(rps_curve, label='RPS (with Bursts)')
        plt.title('Requests Per Second')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(ratio_curve, color='orange', label='Code Request Ratio')
        plt.title('Workload Mixture (Prompt-Heavy vs Token-Heavy)')
        plt.legend()
        plt.tight_layout()
        plt.savefig("notebooks/trace/plot/notrace_preview_v2.png")
        print("Saved trace preview to trace_preview.png")

        # 2. 离散化生成请求
        all_requests = []
        global_req_id = 0
        
        # 预加载数据采样，提高性能
        # 估算总请求数
        total_estimated_reqs = int(rps_curve.sum() * 1.1)
        
        print(f"Sampling requests from datasets (Est. total: {total_estimated_reqs})...")
        
        # 向量化生成每秒的请求数 (非齐次泊松过程近似)
        # N_t ~ Poisson(lambda_t)
        requests_per_sec = np.random.poisson(rps_curve)
        
        # 向量化生成每一秒的类型选择
        # Binomial(n, p)
        code_counts = np.random.binomial(requests_per_sec, ratio_curve)
        conv_counts = requests_per_sec - code_counts
        
        total_code = code_counts.sum()
        total_conv = conv_counts.sum()
        
        # 批量采样 Prompt/Token Sizes
        code_samples = self.code_df.sample(n=total_code, replace=True)
        conv_samples = self.conv_df.sample(n=total_conv, replace=True)
        
        code_idx = 0
        conv_idx = 0
        
        # 3. 组装 DataFrame
        # 这里为了保持时间顺序，我们按秒遍历
        req_id_list = []
        arr_time_list = []
        p_size_list = []
        t_size_list = []
        
        print("Assembling trace...")
        for t in range(duration_sec):
            n_code = code_counts[t]
            n_conv = conv_counts[t]
            n_total = n_code + n_conv
            
            if n_total == 0:
                continue
                
            # 生成这一秒内的具体到达时间 (均匀分布)
            # t + random_offset
            offsets = np.random.uniform(0, 1, n_total)
            offsets.sort() # 排序保证时间递增
            arrival_times = t + offsets
            
            # 提取数据
            # Code 部分
            c_slice = code_samples.iloc[code_idx : code_idx + n_code]
            code_idx += n_code
            
            # Conv 部分
            v_slice = conv_samples.iloc[conv_idx : conv_idx + n_conv]
            conv_idx += n_conv
            
            # 合并当前秒的 Prompt/Token 大小
            # 注意：我们需要打乱顺序，否则前半秒全是 Code，后半秒全是 Conv
            # 虽然对仿真影响不大，但打乱更真实
            p_sizes = np.concatenate([c_slice['ContextTokens'].values, v_slice['ContextTokens'].values])
            t_sizes = np.concatenate([c_slice['GeneratedTokens'].values, v_slice['GeneratedTokens'].values])
            
            # 随机打乱当前秒的请求属性顺序
            perm = np.random.permutation(n_total)
            p_sizes = p_sizes[perm]
            t_sizes = t_sizes[perm]
            
            # 添加到列表
            start_id = global_req_id
            req_ids = np.arange(start_id, start_id + n_total)
            global_req_id += n_total
            
            req_id_list.append(req_ids)
            arr_time_list.append(arrival_times)
            p_size_list.append(p_sizes)
            t_size_list.append(t_sizes)
            
        # 4. 构建最终 DataFrame
        trace_df = pd.DataFrame({
            "request_id": np.concatenate(req_id_list),
            "request_type": 2, # LLM Inference
            "application_id": 0,
            "arrival_timestamp": np.concatenate(arr_time_list),
            "batch_size": 1,
            "prompt_size": np.concatenate(p_size_list).astype(int),
            "token_size": np.concatenate(t_size_list).astype(int)
        })
        
        # 保存
        trace_df.to_csv(output_file, index=False)
        print(f"Trace generation complete! Saved to {output_file}")
        print(f"Total Requests: {len(trace_df)}")
        print(f"Average RPS: {len(trace_df)/duration_sec:.2f}")

if __name__ == "__main__":
    # 1. 确保数据存在
    download_azure_llm_traces()
    
    # 2. 初始化生成器
    gen = SyntheticTraceGenerator(
        code_file="data/code_distributions.csv",
        conv_file="data/conv_distributions.csv",
        seed=2024 # 设置种子保证复现
    )
    
    # 3. 生成 4小时 (14400秒) 的高强度训练数据
    # 这包含了 120 次突发，且混合了不同的负载类型
    gen.generate_trace(duration_sec=14400, output_file="traces/rl_training_mixed_v2.csv")