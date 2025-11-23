import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from utils import get_summary_data, get_request_data
from perf_model import PerfModel

# 设置全局字体大小，比默认字体小2号
plt.rcParams.update({
    'font.size': 8,  # 默认字体大小减小2号
    'axes.titlesize': 10,  # 标题字体大小
    'axes.labelsize': 8,   # 轴标签字体大小
    'xtick.labelsize': 6,  # x轴刻度字体大小
    'ytick.labelsize': 6,  # y轴刻度字体大小
    'legend.fontsize': 6,  # 图例字体大小
    'figure.titlesize': 12 # 图形标题字体大小
})

results_dir = "../results"
plots_dir = "../plots/new"
perf_model_path = "../data/perf_model.csv"

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
            request_df["baseline_e2e"] = request_df["baseline_ttft"] + request_df["baseline_tbt"] * (request_df["token_sizes"] - 1)
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
            for quantile in quantiles:
                result[f"ttft_slowdown_p{int(quantile * 100)}"] = request_df["ttft_slowdown"].quantile(quantile)
                result[f"tbt_slowdown_p{int(quantile * 100)}"] = request_df["tbt_slowdown"].quantile(quantile)
                result[f"e2e_slowdown_p{int(quantile * 100)}"] = request_df["e2e_slowdown"].quantile(quantile)
            for quantile in quantiles:
                result[f"ttft_times_p{int(quantile * 100)}"] = summary_df[f"ttft_times_p{int(quantile * 100)}"][0]
                result[f"tbt_times_p{int(quantile * 100)}"] = summary_df[f"tbt_times_p{int(quantile * 100)}"][0]
                result[f"e2e_times_p{int(quantile * 100)}"] = summary_df[f"response_times_p{int(quantile * 100)}"][0]
                # 添加nth_token_overheads和queue_times的统计信息
                result[f"nth_token_overheads_p{int(quantile * 100)}"] = request_df["nth_token_overheads"].quantile(quantile)
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
    # if quantile == 0.5 or quantile == 0.9:
    #     return {
    #         'bottom': 0,
    #         'top': 4
    #     }
    # elif quantile == 0.99:
    #     return {
    #         'bottom': 0,
    #         'top': 8
    #     }
    return {
            'bottom': 0,
            'top': 8
        }
    raise Exception(f"Invalid y_var:quantile {y_var}:{quantile}")   

def plot_y_vs_trace_new(results_df,
                       traces,
                       y_vars=["ttft_times", "tbt_times", "e2e_times"],
                       y_vars_labels=["TTFT", "TBT", "E2E"],
                       quantiles=[0.5, 0.9, 0.99],
                       title=None):
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
                f"Normalized\np{int(quantile*100)} {y_vars_labels[y_vars.index(y_var)]}")
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

def plot_additional_metrics(results_df,
                           traces,
                           y_vars=["nth_token_overheads", "queue_times"],
                           y_vars_labels=["Nth Token Overheads", "Queue Times"],
                           quantiles=[0.5, 0.9, 0.99],
                           title=None):
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
                f"Time (s)\np{int(quantile*100)} {y_vars_labels[y_vars.index(y_var)]}")
            # 设置y轴范围
            axs[y_vars.index(y_var)][quantiles.index(quantile)].set_ylim(bottom=0)
            
    if title:
        fig.suptitle(title)

    plt.margins(x=0)
    # Set 300dpi
    plt.gcf().set_dpi(300)

def main():
    """
    Main function to generate plots for adaptive_mixed_pool and mixed_pool configurations.
    """
    # Define configurations for adaptive_mixed_pool and mixed_pool
    adaptive_mixed_pool_config = {
        "name": "new_method",
        "scheduler": "adaptive_pool",
        "start_state": "splitwise_25_15",
        "cluster": "0_40"
    }
    
    mixed_pool_config = {
        "name": "splitwise",
        "scheduler": "mixed_pool",
        "start_state": "splitwise_25_15",
        "cluster": "0_40"
    }
    
    configs = [adaptive_mixed_pool_config, mixed_pool_config]
    
    # Define traces for different loads (rr_code_x where x varies)
    traces_index = 0
    traces_name =['conv','code']
    traces = [f"rr_{traces_name[traces_index]}_{i}" for i in range(30, 160, 10)]  # Example range
    
    # Get data
    results_df, request_dfs = get_data(configs, traces, seed=0, model="bloom-176b")
    
    # Generate plots for slowdown metrics
    plot_y_vs_trace_new(
        results_df,
        traces,
        y_vars=["ttft_slowdown", "tbt_slowdown", "e2e_slowdown"],
        y_vars_labels=["TTFT", "TBT", "E2E"],
        title=None
    )
    
    # Save plot
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f"{plots_dir}/TTFT-TBT-{traces_name[traces_index]}-new.png", bbox_inches='tight')
    
    # Generate plots for additional metrics (nth_token_overheads and queue_times)
    plot_additional_metrics(
        results_df,
        traces,
        y_vars=["nth_token_overheads", "queue_times"],
        y_vars_labels=["Nth Token Overheads", "Queue Times"],
        title=None
    )
    
    # Save additional metrics plot
    plt.savefig(f"{plots_dir}/additional_metrics_{traces_name[traces_index]}-new.png", bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    main()