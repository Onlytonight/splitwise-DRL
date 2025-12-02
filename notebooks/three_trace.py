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
    'axes.labelsize': 8,  # 轴标签字体大小
    'xtick.labelsize': 6,  # x轴刻度字体大小
    'ytick.labelsize': 6,  # y轴刻度字体大小
    'legend.fontsize': 6,  # 图例字体大小
    'figure.titlesize': 12  # 图形标题字体大小
})

results_dir = "../results"
plots_dir = "../plots/Unified/"
perf_model_path = "../data/perf_model.csv"
os.makedirs(plots_dir, exist_ok=True)


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


def plot_three_traces(results_df,
                      traces_conv,
                      traces_code,
                      traces_mixed,
                      y_vars=["ttft_times", "tbt_times", "e2e_times"],
                      y_vars_labels=["TTFT", "TBT", "E2E"],
                      quantiles=[0.5, 0.9, 0.99],
                      title=None,
                      save_path=None):
    """
    Create plots showing performance metrics for three different trace types on the same chart.
    """
    fig, axs = plt.subplots(nrows=len(y_vars),
                            ncols=len(quantiles),
                            figsize=(len(quantiles) * 2.5, len(y_vars) * 1.5),
                            sharex=True,
                            constrained_layout=True)

    # Add trace type information to the dataframe
    results_df["trace_type"] = "unknown"
    for idx, row in results_df.iterrows():
        trace = row["trace"]
        if trace in traces_conv:
            results_df.at[idx, "trace_type"] = "conv"
        elif trace in traces_code:
            results_df.at[idx, "trace_type"] = "code"
        elif trace in traces_mixed:
            results_df.at[idx, "trace_type"] = "mixed"

    # Map trace names to load values for x-axis
    def extract_load(trace_name):
        if "mixed_qps" in trace_name:
            return int(trace_name.split("_")[2])  # mixed_qps_{i}_code30
        else:
            return int(trace_name.split("_")[-1])  # rr_conv_i or rr_code_i

    results_df["load"] = results_df["trace"].apply(extract_load)

    # Sort by load for proper line plotting
    results_df = results_df.sort_values("load")

    # Plot each trace type with a different color/style
    trace_types = ["conv", "code", "mixed"]
    trace_labels = ["CONV", "CODE", "MIXED"]
    colors = ["blue", "orange", "green"]

    for y_var in y_vars:
        for quantile in quantiles:
            ax = axs[y_vars.index(y_var)][quantiles.index(quantile)]

            for i, trace_type in enumerate(trace_types):
                filtered_df = results_df[results_df["trace_type"] == trace_type]
                if not filtered_df.empty:
                    ax.plot(filtered_df["load"],
                            filtered_df[f"{y_var}_p{int(quantile * 100)}"],
                            marker='o',
                            markersize=4,
                            label=trace_labels[i],
                            color=colors[i])

            ax.set_xlabel("Request Rate (req/s)")
            ax.set_ylabel(f"Normalized\np{int(quantile * 100)} {y_vars_labels[y_vars.index(y_var)]}")
            ax.grid(True)
            ax.legend()

            # Set y-axis limits
            y_limits = get_y_limits(y_var, quantile)
            ax.set_ylim(**y_limits)

            # Add SLO lines
            slo = get_slo(y_var, quantile)
            ax.axhline(y=slo, color="red", linestyle="--", alpha=0.7)

            # Set x-axis ticks
            all_loads = sorted(results_df["load"].unique())
            ax.set_xticks(all_loads[::2])  # Show every 2nd tick to avoid crowding

    if title:
        fig.suptitle(title)

    plt.margins(x=0)
    # Set 300dpi
    plt.gcf().set_dpi(300)
    plt.savefig(save_path + "-three-traces.png", bbox_inches='tight')


def plot_additional_metrics_three_traces(results_df,
                                         traces_conv,
                                         traces_code,
                                         traces_mixed,
                                         y_vars=["nth_token_overheads", "queue_times"],
                                         y_vars_labels=["Nth Token Overheads", "Queue Times"],
                                         quantiles=[0.5, 0.9, 0.99],
                                         title=None,
                                         save_path=None):
    """
    Create plots for additional metrics for three different trace types on the same chart.
    """
    fig, axs = plt.subplots(nrows=len(y_vars),
                            ncols=len(quantiles),
                            figsize=(len(quantiles) * 2.5, len(y_vars) * 1.5),
                            sharex=True,
                            constrained_layout=True)

    # Add trace type information to the dataframe
    results_df["trace_type"] = "unknown"
    for idx, row in results_df.iterrows():
        trace = row["trace"]
        if trace in traces_conv:
            results_df.at[idx, "trace_type"] = "conv"
        elif trace in traces_code:
            results_df.at[idx, "trace_type"] = "code"
        elif trace in traces_mixed:
            results_df.at[idx, "trace_type"] = "mixed"

    # Map trace names to load values for x-axis
    def extract_load(trace_name):
        if "mixed_qps" in trace_name:
            return int(trace_name.split("_")[2])  # mixed_qps_{i}_code30
        else:
            return int(trace_name.split("_")[-1])  # rr_conv_i or rr_code_i

    results_df["load"] = results_df["trace"].apply(extract_load)

    # Sort by load for proper line plotting
    results_df = results_df.sort_values("load")

    # Plot each trace type with a different color/style
    trace_types = ["conv", "code", "mixed"]
    trace_labels = ["CONV", "CODE", "MIXED"]
    colors = ["blue", "orange", "green"]

    for y_var in y_vars:
        for quantile in quantiles:
            ax = axs[y_vars.index(y_var)][quantiles.index(quantile)]

            for i, trace_type in enumerate(trace_types):
                filtered_df = results_df[results_df["trace_type"] == trace_type]
                if not filtered_df.empty:
                    ax.plot(filtered_df["load"],
                            filtered_df[f"{y_var}_p{int(quantile * 100)}"],
                            marker='o',
                            markersize=4,
                            label=trace_labels[i],
                            color=colors[i])

            ax.set_xlabel("Request Rate (req/s)")
            ax.set_ylabel(f"Time (s)\np{int(quantile * 100)} {y_vars_labels[y_vars.index(y_var)]}")
            ax.grid(True)
            ax.legend()

            # Set y-axis limits (bottom=0)
            ax.set_ylim(bottom=0)

            # Set x-axis ticks
            all_loads = sorted(results_df["load"].unique())
            ax.set_xticks(all_loads[::2])  # Show every 2nd tick to avoid crowding

    if title:
        fig.suptitle(title)

    plt.margins(x=0)
    # Set 300dpi
    plt.gcf().set_dpi(300)
    plt.savefig(save_path + "-additional-three-traces.png", bbox_inches='tight')


def main():
    """
    Main function to generate plots for three different trace types.
    """
    # Define configurations
    mixed_pool_config = {
        "name": "splitwise",
        "scheduler": "mixed_pool",
        "start_state": "splitwise_25_15",
        "cluster": "0_40"
    }

    configs = [mixed_pool_config]

    # Define traces for different loads
    loads = list(range(30, 150, 10))  # range(30, 150, 10)

    traces_conv = [f"rr_conv_{i}" for i in loads]
    traces_code = [f"rr_code_{i}" for i in loads]
    traces_mixed = [f"mixed_qps_{i}_code30" for i in loads]

    # Combine all traces
    all_traces = traces_conv + traces_code + traces_mixed

    # Get data
    results_df, request_dfs = get_data(configs, all_traces, seed=0, model="bloom-176b")

    name = "splitwise-three-traces"

    # Generate plots for slowdown metrics for three trace types
    plot_three_traces(
        results_df,
        traces_conv,
        traces_code,
        traces_mixed,
        y_vars=["ttft_slowdown", "tbt_slowdown", "e2e_slowdown"],
        y_vars_labels=["TTFT", "TBT", "E2E"],
        title=None,
        save_path=plots_dir + name
    )

    # Generate plots for additional metrics for three trace types
    plot_additional_metrics_three_traces(
        results_df,
        traces_conv,
        traces_code,
        traces_mixed,
        y_vars=["nth_token_overheads", "queue_times"],
        y_vars_labels=["Nth Token Overheads", "Queue Times"],
        title=None,
        save_path=plots_dir + name
    )


if __name__ == "__main__":
    main()
