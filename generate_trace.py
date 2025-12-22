import os

from collections import namedtuple

import requests

import numpy as np
import pandas as pd

from scipy import stats




Distributions = namedtuple('Distributions', ['application_id',
                                             'request_type',
                                             'arrival_process',
                                             'batch_size',
                                             'prompt_size',
                                             'token_size'])
Distribution = namedtuple('Distribution', ['name', 'params'])


def generate_samples(distribution, params, size):
    """
    Generate random samples from the given distribution.
    """
    if distribution == "constant":
        return np.ones(size) * params["value"]
    elif distribution == "normal":
        return stats.norm(**params).rvs(size=size)
    elif distribution == "truncnorm":
        return stats.truncnorm(**params).rvs(size=size)
    elif distribution == "randint":
        return stats.uniform(**params).rvs(size=size)
    elif distribution == "uniform":
        return stats.uniform(**params).rvs(size=size)
    elif distribution == "exponential":
        return stats.expon(**params).rvs(size=size)
    elif distribution == "poisson":
        return stats.poisson(**params).rvs(size=size)
    elif distribution == "trace":
        df = pd.read_csv(params["filename"])
        return df[params["column"]].sample(size, replace=True).values
    else:
        raise ValueError(f"Invalid distribution: {distribution}")


def generate_trace(max_requests, distributions, end_time=None):
    """
    根据给定的分布生成请求轨迹数据
    
    参数:
    max_requests: 最大请求数量，决定生成多少个请求记录
    distributions: 包含各种参数分布的命名元组，定义了各类参数的生成规则
    end_time: 结束时间（可选），只保留到达时间小于此值的请求
    
    返回:
    trace_df: 包含所有生成请求信息的pandas DataFrame
    """
    # 生成请求ID序列，从0到max_requests-1
    request_ids = np.arange(max_requests)

    # 根据到达过程分布生成到达时间间隔，并计算累积和得到绝对到达时间
    arrival_timestamps = generate_samples(distributions.arrival_process.name,
                                          distributions.arrival_process.params,
                                          max_requests)
    # 计算累积到达时间（当前请求的到达时间 = 前一个请求到达时间 + 当前请求间隔时间）
    arrival_timestamps = np.cumsum(arrival_timestamps)
    
    # 生成应用ID，根据应用ID分布进行采样
    application_ids = generate_samples(distributions.application_id.name,
                                       distributions.application_id.params,
                                       max_requests)
    # 将应用ID转换为整数类型
    application_ids = map(int, application_ids)
    
    # 生成批处理大小，根据批处理大小分布进行采样
    batch_sizes = generate_samples(distributions.batch_size.name,
                                   distributions.batch_size.params,
                                   max_requests)
    # 将批处理大小转换为整数类型
    batch_sizes = map(int, batch_sizes)
    
    # 生成提示词大小，根据提示词大小分布进行采样
    prompt_sizes = generate_samples(distributions.prompt_size.name,
                                    distributions.prompt_size.params,
                                    max_requests)
    # 将提示词大小转换为整数类型
    prompt_sizes = map(int, prompt_sizes)
    
    # 生成令牌大小，根据令牌大小分布进行采样
    token_sizes = generate_samples(distributions.token_size.name,
                                   distributions.token_size.params,
                                   max_requests)
    # 将令牌大小转换为整数类型
    token_sizes = map(int, token_sizes)
    
    # 生成请求类型ID，根据请求类型分布进行采样
    request_type_ids = generate_samples(distributions.request_type.name,
                                        distributions.request_type.params,
                                        max_requests)
    # 将请求类型ID转换为整数类型
    request_type_ids = map(int, request_type_ids)

    # 将所有生成的数据组合成一个DataFrame
    trace_df = pd.DataFrame({
        "request_id": request_ids,              # 请求唯一标识符
        "request_type": request_type_ids,       # 请求类型（如LLM推理）
        "application_id": application_ids,      # 应用程序标识符
        "arrival_timestamp": arrival_timestamps, # 请求到达时间戳（累积）
        "batch_size": batch_sizes,              # 批处理大小
        "prompt_size": prompt_sizes,            # 提示词大小（token数量）
        "token_size": token_sizes,              # 生成令牌大小（token数量）
    })

    # 如果指定了结束时间，则过滤掉超过该时间的请求
    if end_time is not None:
        trace_df = trace_df[trace_df["arrival_timestamp"] < end_time]

    return trace_df


def get_exponential_scale(num_servers, utilization, request_duration):
    """
    assumes that request_duration is in seconds
    """
    interarrival_time = request_duration / (1.0 * utilization)
    exponential_scale = interarrival_time / num_servers
    return exponential_scale


def generate_trace_from_utilization(
    max_requests,
    end_time,
    num_servers,
    utilization,
    request_duration,
    pt_distributions_file):
    """
    Generate request traces for the simulator using prompt and token
    size distributions.
    """
    exponential_scale = get_exponential_scale(num_servers, utilization, request_duration)
    distributions = Distributions(
        application_id=Distribution("constant", {"value": 0}),
        request_type=Distribution("constant", {"value": 2}), # 2 is for LLM inference
        arrival_process=Distribution("exponential", {"scale": exponential_scale}),
        prompt_size=Distribution("trace", {"filename": pt_distributions_file,
                                           "column": "ContextTokens"}),
        token_size=Distribution("trace", {"filename": pt_distributions_file,
                                          "column": "GeneratedTokens"}),
        batch_size=Distribution("constant", {"value": 1}),
    )

    trace_df = generate_trace(max_requests,
                              distributions,
                              end_time=end_time)
    return trace_df


def generate_trace_from_prompt_token_size_distributions(
    max_requests,
    end_time,
    request_rate,
    pt_distributions_filename):
    """
    根据提示词(prompt)和生成令牌(token)大小分布生成请求轨迹(trace)，用于模拟器仿真
    
    参数:
    max_requests: 最大请求数量
    end_time: 结束时间（秒）
    request_rate: 请求速率（每秒请求数）
    pt_distributions_filename: 包含提示词和令牌大小分布数据的CSV文件路径
    """
    # 定义各种参数的分布类型和参数值
    distributions = Distributions(
        # 应用ID固定为0
        application_id=Distribution("constant", {"value": 0}),
        # 请求类型固定为2（代表LLM推理）
        request_type=Distribution("constant", {"value": 2}), 
        # 到达过程使用指数分布，scale参数是请求间隔的平均时间（1/请求率）
        arrival_process=Distribution("exponential", {"scale": 1.0 / request_rate}),
        # 提示词大小从CSV文件的"ContextTokens"列中采样
        prompt_size=Distribution("trace", {"filename": pt_distributions_filename,
                                           "column": "ContextTokens"}),
        # 下面是被注释掉的截断正态分布替代方案
        #prompt_size=Distribution("truncnorm", {"a": (prompt_min-prompt_mean)/prompt_std,
        #                                       "b": (prompt_max-prompt_mean)/prompt_std,
        #                                       "loc": prompt_mean,
        #                                       "scale": prompt_std}),
        # 生成令牌大小从CSV文件的"GeneratedTokens"列中采样
        token_size=Distribution("trace", {"filename": pt_distributions_filename,
                                          "column": "GeneratedTokens"}),
        # 下面是被注释掉的截断正态分布替代方案
        #token_size=Distribution("truncnorm", {"a": (token_min-token_mean)/token_std,
        #                                      "b": (token_max-token_mean)/token_std,
        #                                      "loc": token_mean,
        #                                      "scale": token_std}),
        # 批处理大小固定为1
        batch_size=Distribution("constant", {"value": 1}),
    )
    # 调用generate_trace函数生成轨迹DataFrame
    trace_df = generate_trace(max_requests,
                              distributions,
                              end_time=end_time)
    return trace_df


def generate_traces(max_requests,
                    end_time,
                    request_rates,
                    pt_distributions_file,
                    trace_filename_template):
    """
    Generate traces with prompt/token size distributions.
    """
    for request_rate in request_rates:
        trace_df = generate_trace_from_prompt_token_size_distributions(
            max_requests,
            end_time,
            request_rate,
            pt_distributions_file)
        trace_filename = trace_filename_template.format(request_rate)
        trace_df.to_csv(trace_filename, index=False)


def generate_code_traces(
    max_requests,
    end_time,
    request_rates,
    code_distributions_file,
    trace_filename_template="traces/rr_code_{}.csv"):
    """
    code traces distribution
    prompt_mean = 2048, prompt_std = 1973, prompt_min = 3, prompt_max = 7437
    token_mean = 28, token_std = 60, token_min = 6, token_max = 1899
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]):
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])

    generate_traces(max_requests,
                    end_time,
                    request_rates,
                    code_distributions_file,
                    trace_filename_template)


def generate_conv_traces(
    max_requests,
    end_time,
    request_rates,
    conv_distributions_file,
    trace_filename_template="traces/rr_conv_{}.csv"):
    """
    conv traces distribution
    prompt_mean = 1155, prompt_std = 1109, prompt_min = 2, prompt_max = 14050
    token_mean = 211, token_std = 163, token_min = 7, token_max = 1000
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]):
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])

    generate_traces(max_requests,
                    end_time,
                    request_rates,
                    conv_distributions_file,
                    trace_filename_template)


def download_file(url, filename):
    """
    Download a file from the given URL.
    """
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)


def download_azure_llm_traces():
    """
    Download traces from the given URL.
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    url_base = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/"

    if not os.path.exists("data/code_distributions.csv"):
        url = url_base + "AzureLLMInferenceTrace_code.csv"
        download_file(url, "data/code_distributions.csv")
        print("Downloaded code traces")

    if not os.path.exists("data/conv_distributions.csv"):
        url = url_base + "AzureLLMInferenceTrace_conv.csv"
        download_file(url, "data/conv_distributions.csv")
        print("Downloaded conv traces")


if __name__ == "__main__":
    np.random.seed(0)

    # download prompt and token size distributions
    download_azure_llm_traces()
    #
    # # generate request traces
    # generate_code_traces(
    #     max_requests=1000000,
    #     end_time=600,
    #     request_rates=list(range(30, 251, 10)),
    #     code_distributions_file="data/code_distributions.csv")
    # print("Generated code traces")

    generate_conv_traces(
        max_requests=1000000,
        end_time=600,
        request_rates=list(range(10, 501, 10)),
        conv_distributions_file="data/conv_distributions.csv")
    print("Generated conv traces")

    # generate request traces for 2 min
    # generate_code_traces(
    #     max_requests=1000000,
    #     end_time=120,
    #     request_rates=list(range(30, 101, 10)),
    #     code_distributions_file="data/code_distributions.csv",
    #     trace_filename_template="traces/rr_code_{}_2min.csv")
    # print("Generated code 2min traces")
    #
    # generate_conv_traces(
    #     max_requests=1000000,
    #     end_time=120,
    #     request_rates=list(range(30, 101, 10)),
    #     conv_distributions_file="data/conv_distributions.csv",
    #     trace_filename_template="traces/rr_conv_{}_2min.csv")
    # print("Generated conv 2min traces")
