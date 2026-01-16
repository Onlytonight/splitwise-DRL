"""
Motivation Heatmap Visualization - Converted from Jupyter Notebook
Creates heatmaps showing performance metrics (TTFT, TBT, E2E slowdowns) at different quantiles (p50, p90, p99) 
for various prompt and token instance configurations.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add notebooks directory to path to import utility functions
sys.path.append('../notebooks')
from perf_model import PerfModel

# ============================================================================
# Configuration based on motivation.sh
# ============================================================================
results_dir = "../results"
perf_model_path = "../data/perf_model.csv"
plots_dir = "../plots/motivation"
os.makedirs(plots_dir, exist_ok=True)

# Parameters from motivation.sh
seed = 1
trace = "day_30"
scheduler = "mixed_pool"
model = "bloom-176b"
simulator = "baseline_static_pd"
cluster = "0_80"

# Instance configurations to test
prompt_instances = list(range(5, 26))  # 1 to 10
token_instances = list(range(5, 26))   # 1 to 10

# Metrics and quantiles
metrics = ["ttft_slowdown", "tbt_slowdown", "e2e_slowdown"]
metric_labels = ["TTFT Slowdown", "TBT Slowdown", "E2E Slowdown"]
quantiles = [0.5, 0.9, 0.99]
quantile_labels = ["p50", "p90", "p99"]

print(f"Results directory: {results_dir}")
print(f"Trace: {trace}")
print(f"Scheduler: {scheduler}")
print(f"Model: {model}")
print(f"Prompt instances range: {min(prompt_instances)}-{max(prompt_instances)}")
print(f"Token instances range: {min(token_instances)}-{max(token_instances)}")

# ============================================================================
# Function Definitions
# ============================================================================

def load_config_data(prompt_num, token_num):
    """
    Load request data for a specific configuration of prompt and token instances.
    """
    # Try multiple possible paths
    paths = [
        f"{results_dir}/{seed}/splitwise_{prompt_num}_{token_num}/{cluster}/{model}/{scheduler}/{simulator}/detailed/0_trace_1.csv",
        f"{results_dir}/{seed}/splitwise_{prompt_num}_{token_num}/{model}/{scheduler}/detailed/0.csv",
        f"../results/{seed}/splitwise_{prompt_num}_{token_num}/{trace}/{cluster}/{model}/{scheduler}/detailed/0.csv",
    ]
    
    for path in paths:
        try:
            print(path)
            request_df = pd.read_csv(path)
            return request_df, path
        except Exception:
            continue
    
    print(f"  Failed to load ({prompt_num}P, {token_num}T) - tried {len(paths)} paths")
    return None, None


def calculate_slowdown(request_df, perf_model, model, hardware, tp):
    """
    Calculate slowdown metrics following the same approach as generate_plots.py
    """
    # Add baseline performance
    perf_model.add_baseline_perf(request_df, model, hardware, tp)
    
    # Calculate baseline e2e time
    request_df["baseline_e2e"] = request_df["baseline_ttft"] + request_df["baseline_tbt"] * (request_df["token_sizes"] - 1)
    
    # Calculate slowdowns
    request_df["ttft_slowdown"] = request_df["ttft_times"] / request_df["baseline_ttft"]
    request_df["tbt_slowdown"] = request_df["tbt_times"] / request_df["baseline_tbt"]
    request_df["e2e_slowdown"] = request_df["response_times"] / request_df["baseline_e2e"]
    
    return request_df


def get_slo(metric, quantile):
    """Get SLO values for different metrics and quantiles."""
    if metric in ["tbt_slowdown", "e2e_slowdown"]:
        if quantile == 0.5:
            return 1.25
        if quantile == 0.9:
            return 1.5
        if quantile == 0.99:
            return 5
    elif metric == "ttft_slowdown":
        if quantile == 0.5:
            return 2
        if quantile == 0.9:
            return 3
        if quantile == 0.99:
            return 6
    return None


# ============================================================================
# Load Data
# ============================================================================

# Test loading one configuration
print("\nTesting data loading...")
test_df, test_path = load_config_data(10, 10)
if test_df is not None:
    print(f"✓ Successfully loaded test configuration from: {test_path}")
    print(f"  Columns: {test_df.columns.tolist()}")
    print(f"  Shape: {test_df.shape}")
else:
    print("✗ Failed to load test configuration")
    print("\n" + "="*70)
    print("ERROR: Cannot find result data files!")
    print("="*70)
    print("\nExpected path structure:")
    print(f"  {results_dir}/{{seed}}/{{trace}}/{{prompt}}_{{token}}/{{model}}/{{scheduler}}/{{policy}}/detailed/0.csv")
    print("\nPlease ensure:")
    print("  1. You have run the motivation.sh script")
    print("  2. The experiments completed successfully")
    print("  3. The detailed results were saved")
    print("\nAlternatively, update the paths in this script to match your results directory structure.")
    print("="*70)
    sys.exit(1)

# Load performance model for baseline calculations
print("\nLoading performance model...")
try:
    perf_model = PerfModel(perf_model_path, init=True)
    print("✓ Performance model loaded")
except Exception as e:
    print(f"✗ Failed to load performance model: {e}")
    print(f"  Expected path: {perf_model_path}")
    sys.exit(1)

# Define baseline configuration for normalization
normalize_model = model
normalize_hardware = "a100-80gb"
normalize_tp = 8

print(f"Baseline configuration: {normalize_model}, {normalize_hardware}, TP={normalize_tp}")

# Test slowdown calculation
if test_df is not None:
    try:
        test_df_with_slowdown = calculate_slowdown(test_df.copy(), perf_model, normalize_model, normalize_hardware, normalize_tp)
        print(f"\n✓ Slowdown calculation successful")
        print(f"  TTFT slowdown mean: {test_df_with_slowdown['ttft_slowdown'].mean():.2f}")
        print(f"  TBT slowdown mean: {test_df_with_slowdown['tbt_slowdown'].mean():.2f}")
        print(f"  E2E slowdown mean: {test_df_with_slowdown['e2e_slowdown'].mean():.2f}")
    except Exception as e:
        print(f"✗ Slowdown calculation failed: {e}")
        sys.exit(1)

# ============================================================================
# Load All Configurations
# ============================================================================

print("\n" + "="*70)
print("Loading all configurations...")
print("="*70)
results = []

total_configs = len(prompt_instances) * len(token_instances)
loaded_count = 0

for prompt_num in prompt_instances:
    for token_num in token_instances:
        request_df, path = load_config_data(prompt_num, token_num)
        
        if request_df is None:
            continue
        
        # Calculate slowdowns
        request_df = calculate_slowdown(request_df, perf_model, normalize_model, normalize_hardware, normalize_tp)
        
        # Extract quantile values for each metric
        result = {
            'prompt_instances': prompt_num,
            'token_instances': token_num
        }
        
        for metric in metrics:
            for quantile in quantiles:
                result[f"{metric}_p{int(quantile * 100)}"] = request_df[metric].quantile(quantile)
        
        results.append(result)
        loaded_count += 1
        if loaded_count % 10 == 0:
            print(f"  Loaded {loaded_count}/{total_configs} configurations...")

print(f"\n✓ Loaded {loaded_count}/{total_configs} configurations")

if loaded_count == 0:
    print("\n✗ No configurations loaded. Cannot proceed with plotting.")
    sys.exit(1)

# Create results DataFrame
results_df = pd.DataFrame(results)
csv_path = os.path.join(plots_dir, "results_df.csv")
results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Results DataFrame shape: {results_df.shape}")
print(f"结果数据已保存至: {csv_path}")

# ============================================================================
# Create Heatmap Matrices
# ============================================================================

print("\n" + "="*70)
print("Creating heatmap matrices...")
print("="*70)
heatmap_data = {}

for metric in metrics:
    for quantile in quantiles:
        column_name = f"{metric}_p{int(quantile * 100)}"
        
        # Create pivot table with prompt_instances as rows and token_instances as columns
        pivot_table = results_df.pivot(
            index='prompt_instances',
            columns='token_instances',
            values=column_name
        )
        
        heatmap_data[column_name] = pivot_table

print(f"✓ Created {len(heatmap_data)} heatmap matrices")
print(f"  Heatmap shape: {list(heatmap_data.values())[0].shape}")

# ============================================================================
# Print SLO Information
# ============================================================================

print("\n" + "="*70)
print("SLO Thresholds")
print("="*70)
for metric in metrics:
    for quantile in quantiles:
        slo = get_slo(metric, quantile)
        print(f"{metric:20s} @ p{int(quantile*100):2d}: SLO = {slo}")

# ============================================================================
# Create Main Visualization: 3x3 Grid
# ============================================================================

print("\n" + "="*70)
print("Generating main heatmap (3x3 grid)...")
print("="*70)

fig, axes = plt.subplots(3, 3, figsize=(18, 16))
fig.suptitle('Performance Heatmaps: Prompt vs Token Instances', 
             fontsize=20, fontweight='bold', y=0.995)

# Configure plot style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

for i, metric in enumerate(metrics):
    for j, quantile in enumerate(quantiles):
        ax = axes[i, j]
        column_name = f"{metric}_p{int(quantile * 100)}"
        data = heatmap_data[column_name]
        
        # Get SLO for this metric/quantile combination
        slo = get_slo(metric, quantile)
        
        # Determine color scale
        vmin = data.min().min()
        vmax = min(data.max().max(), slo * 2 if slo else data.max().max())  # Cap at 2x SLO
        
        # Create heatmap
        sns.heatmap(data, 
                    annot=True, 
                    fmt='.2f',
                    cmap='RdYlGn_r',  # Red for high (bad), Green for low (good)
                    vmin=vmin,
                    vmax=vmax,
                    cbar_kws={'label': 'Slowdown'},
                    ax=ax,
                    linewidths=0.5,
                    linecolor='gray')
        
        # Set labels
        ax.set_xlabel('Token Instances', fontweight='bold')
        ax.set_ylabel('Prompt Instances', fontweight='bold')
        
        # Add title with SLO information
        title = f"{metric_labels[i]} @ {quantile_labels[j]}"
        if slo:
            title += f" (SLO: {slo})"
        ax.set_title(title, fontweight='bold', pad=10)
        
        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.tight_layout()
save_path = f"{plots_dir}/motivation_heatmap.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {save_path}")
plt.close()

# ============================================================================
# Create Individual Metric Heatmaps
# ============================================================================

print("\nGenerating individual metric heatmaps...")
for i, metric in enumerate(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{metric_labels[i]} across Different Quantiles', 
                 fontsize=16, fontweight='bold')
    
    for j, quantile in enumerate(quantiles):
        ax = axes[j]
        column_name = f"{metric}_p{int(quantile * 100)}"
        data = heatmap_data[column_name]
        
        slo = get_slo(metric, quantile)
        vmin = data.min().min()
        vmax = min(data.max().max(), slo * 2 if slo else data.max().max())
        
        sns.heatmap(data, 
                    annot=True, 
                    fmt='.2f',
                    cmap='RdYlGn_r',
                    vmin=vmin,
                    vmax=vmax,
                    cbar_kws={'label': 'Slowdown'},
                    ax=ax,
                    linewidths=0.5,
                    linecolor='gray')
        
        ax.set_xlabel('Token Instances', fontweight='bold')
        ax.set_ylabel('Prompt Instances', fontweight='bold')
        
        title = f"{quantile_labels[j]}"
        if slo:
            title += f" (SLO: {slo})"
        ax.set_title(title, fontweight='bold')
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    save_path = f"{plots_dir}/motivation_{metric}_all_quantiles.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

print("\n✓ All individual metric heatmaps saved")

# ============================================================================
# Find Optimal Configurations
# ============================================================================

print("\n" + "="*70)
print("OPTIMAL CONFIGURATION ANALYSIS")
print("="*70)

# Mark configurations that meet all SLOs
results_df['meets_all_slo'] = True

for metric in metrics:
    for quantile in quantiles:
        slo = get_slo(metric, quantile)
        if slo:
            column_name = f"{metric}_p{int(quantile * 100)}"
            results_df['meets_all_slo'] &= (results_df[column_name] <= slo)

configs_within_slo = results_df[results_df['meets_all_slo']]

print(f"\nConfigurations meeting ALL SLOs: {len(configs_within_slo)}/{len(results_df)}")

if len(configs_within_slo) > 0:
    print("\nConfigurations within SLO:")
    print(configs_within_slo[['prompt_instances', 'token_instances']].to_string(index=False))
    
    # Find the configuration with minimum total instances
    configs_within_slo['total_instances'] = configs_within_slo['prompt_instances'] + configs_within_slo['token_instances']
    optimal = configs_within_slo[configs_within_slo['total_instances'] == configs_within_slo['total_instances'].min()]
    
    print(f"\n✓ Optimal configuration(s) (minimum total instances):")
    for _, row in optimal.iterrows():
        print(f"  Prompt instances: {int(row['prompt_instances'])}, Token instances: {int(row['token_instances'])}, Total: {int(row['total_instances'])}")
else:
    print("\n⚠ No configuration meets all SLOs")
    print("\nClosest configurations:")
    # Count how many SLOs each config violates
    results_df['slo_violations'] = 0
    for metric in metrics:
        for quantile in quantiles:
            slo = get_slo(metric, quantile)
            if slo:
                column_name = f"{metric}_p{int(quantile * 100)}"
                results_df['slo_violations'] += (results_df[column_name] > slo).astype(int)
    
    best_configs = results_df.nsmallest(5, 'slo_violations')
    print(best_configs[['prompt_instances', 'token_instances', 'slo_violations']].to_string(index=False))

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

for metric in metrics:
    print(f"\n{metric.upper().replace('_', ' ')}:")
    for quantile in quantiles:
        column_name = f"{metric}_p{int(quantile * 100)}"
        slo = get_slo(metric, quantile)
        
        values = results_df[column_name]
        print(f"  {quantile_labels[quantiles.index(quantile)]}:")
        print(f"    Min: {values.min():.3f}")
        print(f"    Max: {values.max():.3f}")
        print(f"    Mean: {values.mean():.3f}")
        print(f"    Median: {values.median():.3f}")
        if slo:
            configs_meeting_slo = (values <= slo).sum()
            print(f"    SLO ({slo}): {configs_meeting_slo}/{len(results_df)} configs meeting SLO")

print("\n" + "="*70)
print(f"Total configurations analyzed: {len(results_df)}")
print(f"All plots saved to: {plots_dir}/")
print("="*70)
print("\n✓ Script completed successfully!")

