import os
import sys
import numpy as np
import pandas as pd

# Add the parent directory to the path so we can import generate_trace
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from generate_trace import (
    generate_trace_from_prompt_token_size_distributions,
    download_azure_llm_traces
)


def generate_periodic_burst_trace(
        base_request_rate,
        burst_multiplier,
        period_duration,
        burst_duration,
        cycles,
        pt_distributions_file,
        output_file):
    """
    Generate traces with periodic bursts.

    Each cycle:
    - Duration: period_duration seconds
    - Base request rate for the entire period
    - Burst request rate for burst_duration seconds in the middle of the period
    - Exactly same pattern repeated for all cycles

    Args:
        base_request_rate: Base requests per second during non-burst periods
        burst_multiplier: Multiplier for request rate during burst (e.g., 2.0 means 2x the base rate)
        period_duration: Duration of each cycle in seconds
        burst_duration: Duration of burst within each cycle in seconds
        cycles: Number of cycles to repeat
        pt_distributions_file: Path to the prompt/token size distribution file
        output_file: Output CSV file path
    """

    all_dfs = []
    current_time_offset = 0

    # Calculate when the burst starts within each period
    burst_start_offset = (period_duration - burst_duration) / 2

    print(f"Generating {cycles} cycles of periodic burst traces")
    print(f"Each cycle: {period_duration}s total, burst of {burst_duration}s starting at {burst_start_offset}s")
    print(f"Base rate: {base_request_rate} RPS, burst rate: {base_request_rate * burst_multiplier} RPS")

    for cycle in range(cycles):
        # Non-burst period before the burst
        if burst_start_offset > 0:
            trace_df = generate_trace_from_prompt_token_size_distributions(
                max_requests=int(base_request_rate * burst_start_offset * 2),  # Estimate max requests needed
                end_time=burst_start_offset,
                request_rate=base_request_rate,
                pt_distributions_filename=pt_distributions_file
            )

            # Adjust timestamps
            trace_df['arrival_timestamp'] += current_time_offset
            all_dfs.append(trace_df)
            current_time_offset += burst_start_offset

        # Burst period
        trace_df = generate_trace_from_prompt_token_size_distributions(
            max_requests=int(base_request_rate * burst_multiplier * burst_duration * 2),  # Estimate max requests needed
            end_time=burst_duration,
            request_rate=base_request_rate * burst_multiplier,
            pt_distributions_filename=pt_distributions_file
        )

        # Adjust timestamps
        trace_df['arrival_timestamp'] += current_time_offset
        all_dfs.append(trace_df)
        current_time_offset += burst_duration

        # Non-burst period after the burst
        post_burst_duration = period_duration - burst_start_offset - burst_duration
        if post_burst_duration > 0:
            trace_df = generate_trace_from_prompt_token_size_distributions(
                max_requests=int(base_request_rate * post_burst_duration * 2),  # Estimate max requests needed
                end_time=post_burst_duration,
                request_rate=base_request_rate,
                pt_distributions_filename=pt_distributions_file
            )

            # Adjust timestamps
            trace_df['arrival_timestamp'] += current_time_offset
            all_dfs.append(trace_df)
            current_time_offset += post_burst_duration

        # Print progress
        if (cycle + 1) % 10 == 0:
            print(f"Completed cycle {cycle + 1}/{cycles}")

    # Concatenate all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Sort by arrival timestamp
    combined_df = combined_df.sort_values('arrival_timestamp').reset_index(drop=True)

    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Periodic burst trace saved to {output_file} with {len(combined_df)} total requests")
    print(f"Total simulation time: {current_time_offset} seconds")

    return combined_df


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility

    # Create traces directory if it doesn't exist
    traces_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'traces')
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)

    # Download Azure LLM traces if needed
    download_azure_llm_traces()

    # Generate periodic burst trace
    # Parameters as specified:
    # - Period: 40s
    # - Burst duration: 10s
    # - Number of cycles: 100
    # - Base request rate: 30 RPS
    # - Burst multiplier: 3.0 (90 RPS during burst)

    generate_periodic_burst_trace(
        base_request_rate=30,
        burst_multiplier=3.0,
        period_duration=40,  # 40s per cycle
        burst_duration=10,  # 10s burst within each cycle
        cycles=100,  # Repeat 100 times
        pt_distributions_file="data/code_distributions.csv",
        output_file=os.path.join(traces_dir, "periodic_burst_trace.csv")
    )

    print("Periodic burst trace generated successfully!")