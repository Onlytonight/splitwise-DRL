import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from utils import  get_summary_data, get_request_data
from perf_model import PerfModel

# 设置全局字体大小，比默认字体小2号
plt.rcParams.update({
    'font.size': 8,  # 默认字体大小减小2号
    'axes.titlesize': 10,  # 标题字体大小
    'axes.labelsize': 8,  # 轴标签字体大小
    'xtick.labelsize': 6,  # x轴刻度字体大小
    'ytick.labelsize': 6,  # y轴刻度字体大小
    'legend.fontsize': 6,  # 图例字体大小
    'figure.titlesize': 12  # 图形标题字体大小
})

results_dir = "../results"
plots_dir = "../plots/mixed_trace/"
perf_model_path = "../data/perf_model.csv"
os.makedirs(plots_dir, exist_ok=True)
req_types = [0,1]

# path = 'D:\homework\网络\论文\LLMshedule\pd分离\splitwise-DRL\plots\splitplot'
# def get_summary_data(results_dir, scheduler, start_state, cluster, trace, seed, model=""):
#     try:
#         summary_df = pd.read_csv(path+"/summary.csv")
#     except Exception as e:
#         print(e)
#         print(f"Failed to read {results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/summary.csv")
#         return None
#     return summary_df
#
# def get_request_data(results_dir, scheduler, start_state, cluster, trace, seed, model=""):
#     try:
#         request_df = pd.read_csv(path+"/detailed/0.csv")
#     except:
#         print(f"Failed to read {results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/detailed/0.csv")
#         return None
#     return request_df

def get_data(configs, traces, seed, quantiles=[0.5, 0.9, 0.99], model=""):
    """
    Load and process data from results directory based on configurations and traces.
    """
    results = []
    request_dfs = {}
    for trace in traces:
        for config in configs:
            name = config["name"]
            scheduler = config["scheduler"]
            start_state = config["start_state"]
            cluster = config["cluster"]

            summary_df = get_summary_data(results_dir, scheduler, start_state, cluster, trace, seed, model=model)
            request_df = get_request_data(results_dir, scheduler, start_state, cluster, trace, seed, model=model)
            if summary_df is None or request_df is None:
                continue

            perf_model = PerfModel(perf_model_path, init=True)
            normalize_model = model
            normalize_hardware = "a100-80gb"
            normalize_tp = 8

            perf_model.add_baseline_perf(request_df, normalize_model, normalize_hardware, normalize_tp)
            request_df["baseline_e2e"] = request_df["baseline_ttft"] + request_df["baseline_tbt"] * (
                        request_df["token_sizes"] - 1)
            request_df["ttft_slowdown"] = request_df["ttft_times"] / request_df["baseline_ttft"]
            request_df["tbt_slowdown"] = request_df["tbt_times"] / request_df["baseline_tbt"]
            request_df["e2e_slowdown"] = request_df["response_times"] / request_df["baseline_e2e"]

            # Check if OOM by existence of oom.csv
            oom = False
            if os.path.exists(f"{results_dir}/{seed}/{start_state}/{trace}/{cluster}/{model}/{scheduler}/oom.csv"):
                oom = True

            result = {}
            for key, value in config.items():
                result[key] = value
            result["trace"] = trace
            result["seed"] = seed

            # 如果存在request_types列，则分别计算code和conv类型的指标
            if 'request_types' in request_df.columns:
                for req_type in req_types:
                    type_filtered_df = request_df[request_df['request_types'] == req_type]
                    if not type_filtered_df.empty:
                        for quantile in quantiles:
                            result[f"{req_type}_ttft_slowdown_p{int(quantile * 100)}"] = type_filtered_df["ttft_slowdown"].quantile(quantile)
                            result[f"{req_type}_tbt_slowdown_p{int(quantile * 100)}"] = type_filtered_df["tbt_slowdown"].quantile(quantile)
                            result[f"{req_type}_e2e_slowdown_p{int(quantile * 100)}"] = type_filtered_df["e2e_slowdown"].quantile(quantile)

                        # 对于时间指标，我们使用原始值而不是slowdown
                        for quantile in quantiles:
                            result[f"{req_type}_ttft_times_p{int(quantile * 100)}"] = type_filtered_df["ttft_times"].quantile(quantile)
                            result[f"{req_type}_tbt_times_p{int(quantile * 100)}"] = type_filtered_df["tbt_times"].quantile(quantile)
                            result[f"{req_type}_e2e_times_p{int(quantile * 100)}"] = type_filtered_df["response_times"].quantile(quantile)
                            result[f"{req_type}_nth_token_overheads_p{int(quantile * 100)}"] = type_filtered_df["nth_token_overheads"].quantile(quantile)
                            result[f"{req_type}_queue_times_p{int(quantile * 100)}"] = type_filtered_df["queue_times"].quantile(quantile)

            # 总体统计数据保持不变
            for quantile in quantiles:
                result[f"ttft_slowdown_p{int(quantile * 100)}"] = request_df["ttft_slowdown"].quantile(quantile)
                result[f"tbt_slowdown_p{int(quantile * 100)}"] = request_df["tbt_slowdown"].quantile(quantile)
                result[f"e2e_slowdown_p{int(quantile * 100)}"] = request_df["e2e_slowdown"].quantile(quantile)
            for quantile in quantiles:
                result[f"ttft_times_p{int(quantile * 100)}"] = summary_df[f"ttft_times_p{int(quantile * 100)}"][0]
                result[f"tbt_times_p{int(quantile * 100)}"] = summary_df[f"tbt_times_p{int(quantile * 100)}"][0]
                result[f"e2e_times_p{int(quantile * 100)}"] = summary_df[f"response_times_p{int(quantile * 100)}"][0]
                # 添加nth_token_overheads和queue_times的统计信息
                result[f"nth_token_overheads_p{int(quantile * 100)}"] = request_df["nth_token_overheads"].quantile(
                    quantile)
                result[f"queue_times_p{int(quantile * 100)}"] = request_df["queue_times"].quantile(quantile)
            result["oom"] = oom

            # Save results to later create a dataframe
            results.append(result)
            request_dfs[f"{name}_{trace}"] = request_df

    results_df = pd.DataFrame(results)
    return results_df, request_dfs


def get_slo(y_var, quantile):
    """
    Get SLO values for different metrics and quantiles.
    """
    if y_var == "tbt_slowdown" or y_var == "e2e_slowdown":
        if quantile == 0.5:
            return 1.25
        if quantile == 0.9:
            return 1.5
        if quantile == 0.99:
            return 5
    elif y_var == "ttft_slowdown":
        if quantile == 0.5:
            return 2
        if quantile == 0.9:
            return 3
        if quantile == 0.99:
            return 6
    else:
        raise Exception(f"Invalid y_var:quantile {y_var}:{quantile}")


def get_y_limits(y_var, quantile):
    """
    Get Y-axis limits for plots.
    """
    if quantile == 0.5 or quantile == 0.9:
        return {
            'bottom': 0,
            'top': 4
        }
    elif quantile == 0.99:
        return {
            'bottom': 0,
            'top': 50
        }
    # return {
    #     'bottom': 0,
    #     'top': 8
    # }
    raise Exception(f"Invalid y_var:quantile {y_var}:{quantile}")


def plot_y_vs_trace_new(results_df,
                        traces,
                        y_vars=["ttft_times", "tbt_times", "e2e_times"],
                        y_vars_labels=["TTFT", "TBT", "E2E"],
                        quantiles=[0.5, 0.9, 0.99],
                        title=None,
                        save_path=None):
    """
    Create plots showing performance metrics vs trace/load.
    """
    fig, axs = plt.subplots(nrows=len(y_vars),
                            ncols=len(quantiles),
                            figsize=(len(quantiles) * 2.5, len(y_vars) * 1.5),
                            sharex=True,
                            constrained_layout=True)

    # Plot
    for y_var in y_vars:
        for quantile in quantiles:
            sns.lineplot(data=results_df,
                         x="trace",
                         y=f"{y_var}_p{int(quantile * 100)}",
                         hue="name",
                         style="name",
                         markers=True,
                         markersize=7,
                         ax=axs[y_vars.index(y_var)][quantiles.index(quantile)])

    for ax in axs.flatten():
        ax.grid()
        ax.get_legend().set_visible(False)
        ax.set_xlabel("Request Rate (req/s)")
        xlabels = [trace.split("_")[2] for trace in traces]
        ax.set_xticks(ticks=range(0, len(traces)), labels=xlabels)
        # 解决横坐标数字重叠问题
        ax.tick_params(axis='x', rotation=45)  # 旋转x轴标签
        ax.set_xticklabels(xlabels, rotation=45, ha='right')  # 设置标签旋转和对齐方式

    # Create a single legend in center of figure
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, title="", bbox_to_anchor=(0.5, 1.01))

    for y_var in y_vars:
        for quantile in quantiles:
            axs[y_vars.index(y_var)][quantiles.index(quantile)].set_ylabel(
                f"Normalized\np{int(quantile * 100)} {y_vars_labels[y_vars.index(y_var)]}")
            y_limits = get_y_limits(y_var, quantile)
            axs[y_vars.index(y_var)][quantiles.index(quantile)].set_ylim(**y_limits)
            slo = get_slo(y_var, quantile)
            # Add SLO lines
            axs[y_vars.index(y_var)][quantiles.index(quantile)].axhline(y=slo, color="red", linestyle="--")

    if title:
        fig.suptitle(title)

    plt.margins(x=0)
    # Set 300dpi
    plt.gcf().set_dpi(300)
    plt.savefig(save_path + "-TTFT.png", bbox_inches='tight')


def plot_y_vs_trace_by_request_type(results_df,
                                    traces,
                                    y_vars=["ttft_times", "tbt_times", "e2e_times"],
                                    y_vars_labels=["TTFT", "TBT", "E2E"],
                                    quantiles=[0.5, 0.9, 0.99],
                                    title=None,
                                    save_path=None):
    """
    Create plots showing performance metrics vs trace/load, separated by request types (code/conv).
    """
    fig, axs = plt.subplots(nrows=len(y_vars),
                            ncols=len(quantiles),
                            figsize=(len(quantiles) * 2.5, len(y_vars) * 1.5),
                            sharex=True,
                            constrained_layout=True)

    # 绘制不同请求类型的曲线
    request_types = req_types
    colors = ['blue', 'orange']
    markers = ['o', 's']

    for i, req_type in enumerate(request_types):
        for y_var in y_vars:
            for quantile in quantiles:
                y_col = f"{req_type}_{y_var}_p{int(quantile * 100)}"
                if y_col in results_df.columns:
                    sns.lineplot(data=results_df,
                                 x="trace",
                                 y=y_col,
                                 color=colors[i],
                                 marker=markers[i],
                                 markersize=7,
                                 label=f"{req_type}",
                                 ax=axs[y_vars.index(y_var)][quantiles.index(quantile)])
                else:
                    print(f"Warning: Column {y_col} not found in results_df")

    for ax in axs.flatten():
        ax.grid()
        ax.get_legend().set_visible(False)
        ax.set_xlabel("Request Rate (req/s)")
        xlabels = [trace.split("_")[2] for trace in traces]
        ax.set_xticks(ticks=range(0, len(traces)), labels=xlabels)
        # 解决横坐标数字重叠问题
        ax.tick_params(axis='x', rotation=45)  # 旋转x轴标签
        ax.set_xticklabels(xlabels, rotation=45, ha='right')  # 设置标签旋转和对齐方式

    # Create a single legend in center of figure
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, title="", bbox_to_anchor=(0.5, 1.01))

    for y_var in y_vars:
        for quantile in quantiles:
            axs[y_vars.index(y_var)][quantiles.index(quantile)].set_ylabel(
                f"Time (s)\np{int(quantile * 100)} {y_vars_labels[y_vars.index(y_var)]}")
            y_limits = get_y_limits(y_var, quantile)
            axs[y_vars.index(y_var)][quantiles.index(quantile)].set_ylim(**y_limits)

    if title:
        fig.suptitle(title)

    plt.margins(x=0)
    # Set 300dpi
    plt.gcf().set_dpi(300)
    plt.savefig(save_path + "-by-request-type.png", bbox_inches='tight')


def plot_additional_metrics(results_df,
                            traces,
                            y_vars=["nth_token_overheads", "queue_times"],
                            y_vars_labels=["Nth Token Overheads", "Queue Times"],
                            quantiles=[0.5, 0.9, 0.99],
                            title=None,
                            save_path=None):
    """
    Create plots for additional metrics like nth_token_overheads and queue_times.
    """
    fig, axs = plt.subplots(nrows=len(y_vars),
                            ncols=len(quantiles),
                            figsize=(len(quantiles) * 2.5, len(y_vars) * 1.5),
                            sharex=True,
                            constrained_layout=True)

    # Plot
    for y_var in y_vars:
        for quantile in quantiles:
            sns.lineplot(data=results_df,
                         x="trace",
                         y=f"{y_var}_p{int(quantile * 100)}",
                         hue="name",
                         style="name",
                         markers=True,
                         markersize=7,
                         ax=axs[y_vars.index(y_var)][quantiles.index(quantile)])

    for ax in axs.flatten():
        ax.grid()
        ax.get_legend().set_visible(False)
        ax.set_xlabel("Request Rate (req/s)")
        xlabels = [trace.split("_")[2] for trace in traces]
        ax.set_xticks(ticks=range(0, len(traces)), labels=xlabels)
        # 解决横坐标数字重叠问题
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(xlabels, rotation=45, ha='right')

    # Create a single legend in center of figure
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, title="", bbox_to_anchor=(0.5, 1.01))

    for y_var in y_vars:
        for quantile in quantiles:
            axs[y_vars.index(y_var)][quantiles.index(quantile)].set_ylabel(
                f"Time (s)\np{int(quantile * 100)} {y_vars_labels[y_vars.index(y_var)]}")
            # 设置y轴范围
            axs[y_vars.index(y_var)][quantiles.index(quantile)].set_ylim(bottom=0)

    if title:
        fig.suptitle(title)

    plt.margins(x=0)
    # Set 300dpi
    plt.gcf().set_dpi(300)
    plt.savefig(save_path + ".png", bbox_inches='tight')


def plot_additional_metrics_by_request_type(results_df,
                                            traces,
                                            y_vars=["nth_token_overheads", "queue_times"],
                                            y_vars_labels=["Nth Token Overheads", "Queue Times"],
                                            quantiles=[0.5, 0.9, 0.99],
                                            title=None,
                                            save_path=None):
    """
    Create plots for additional metrics like nth_token_overheads and queue_times, separated by request types (code/conv).
    """
    fig, axs = plt.subplots(nrows=len(y_vars),
                            ncols=len(quantiles),
                            figsize=(len(quantiles) * 2.5, len(y_vars) * 1.5),
                            sharex=True,
                            constrained_layout=True)

    # 绘制不同请求类型的曲线
    request_types = req_types
    colors = ['blue', 'orange']
    markers = ['o', 's']

    for i, req_type in enumerate(request_types):
        for y_var in y_vars:
            for quantile in quantiles:
                y_col = f"{req_type}_{y_var}_p{int(quantile * 100)}"
                if y_col in results_df.columns:
                    sns.lineplot(data=results_df,
                                 x="trace",
                                 y=y_col,
                                 color=colors[i],
                                 marker=markers[i],
                                 markersize=7,
                                 label=f"{req_type}",
                                 ax=axs[y_vars.index(y_var)][quantiles.index(quantile)])
                else:
                    print(f"Warning: Column {y_col} not found in results_df")

    for ax in axs.flatten():
        ax.grid()
        ax.get_legend().set_visible(False)
        ax.set_xlabel("Request Rate (req/s)")
        xlabels = [trace.split("_")[2] for trace in traces]
        ax.set_xticks(ticks=range(0, len(traces)), labels=xlabels)
        # 解决横坐标数字重叠问题
        ax.tick_params(axis='x', rotation=45)
        ax.set_xticklabels(xlabels, rotation=45, ha='right')

    # Create a single legend in center of figure
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, title="", bbox_to_anchor=(0.5, 1.01))

    for y_var in y_vars:
        for quantile in quantiles:
            axs[y_vars.index(y_var)][quantiles.index(quantile)].set_ylabel(
                f"Time (s)\np{int(quantile * 100)} {y_vars_labels[y_vars.index(y_var)]}")
            # 设置y轴范围
            axs[y_vars.index(y_var)][quantiles.index(quantile)].set_ylim(bottom=0)

    if title:
        fig.suptitle(title)

    plt.margins(x=0)
    # Set 300dpi
    plt.gcf().set_dpi(300)
    plt.savefig(save_path + "-additional-by-request-type.png", bbox_inches='tight')


def main():
    """
    Main function to generate plots for adaptive_mixed_pool and mixed_pool configurations.
    """
    # Define configurations for adaptive_mixed_pool and mixed_pool

    mixed_pool_config = {
        "name": "splitwise",
        "scheduler": "mixed_pool",
        "start_state": "splitwise_25_15",
        "cluster": "0_40"
    }

    configs = [mixed_pool_config]
    traces = [f"mixed_qps_{i}_code30" for i in range(30, 141, 10)]  # Example range

    # Get data
    results_df, request_dfs = get_data(configs, traces, seed=0, model="bloom-176b")
    name = 'splitwise'

    # Generate plots for slowdown metrics
    plot_y_vs_trace_new(
        results_df,
        traces,
        y_vars=["ttft_slowdown", "tbt_slowdown", "e2e_slowdown"],
        y_vars_labels=["TTFT", "TBT", "E2E"],
        title=None,
        save_path=plots_dir + name
    )

    # Generate plots for additional metrics (nth_token_overheads and queue_times)
    plot_additional_metrics(
        results_df,
        traces,
        y_vars=["nth_token_overheads", "queue_times"],
        y_vars_labels=["Nth Token Overheads", "Queue Times"],
        title=None,
        save_path=plots_dir + name
    )

    # Generate plots separated by request type
    plot_y_vs_trace_by_request_type(
        results_df,
        traces,
        y_vars=["ttft_slowdown", "tbt_slowdown", "e2e_slowdown"],
        y_vars_labels=["TTFT", "TBT", "E2E"],
        title=None,
        save_path=plots_dir + name
    )

    plot_additional_metrics_by_request_type(
        results_df,
        traces,
        y_vars=["nth_token_overheads", "queue_times"],
        y_vars_labels=["Nth Token Overheads", "Queue Times"],
        title=None,
        save_path=plots_dir + name
    )


if __name__ == "__main__":
    main()
