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

def generate_long_rps_trace(
    max_requests_per_segment,
    segment_duration,
    cycles,
    pt_distributions_file,
    output_file):
    """
    Generate traces with changing request rates over time cycles.
    
    For each cycle:
    - First half: request rates go from 10 to 150
    - Second half: request rates go from 150 to 10
    """
    
    all_dfs = []
    current_time_offset = 0
    
    # Define request rates for ascending and descending phases
    ascending_rates = list(range(10, 101, 10))  # 10, 20, ..., 140, 150
    descending_rates = list(range(100, 9, -10))  # 150, 140, ..., 20, 10
    
    for cycle in range(cycles):
        print(f"Generating cycle {cycle + 1}/{cycles}")
        
        # First half of cycle: ascending rates (10 to 150)
        for i, request_rate in enumerate(ascending_rates):
            # print(f"  Generating segment with rate {request_rate} for {segment_duration}s")
            
            # Generate trace for this request rate and segment
            trace_df = generate_trace_from_prompt_token_size_distributions(
                max_requests=max_requests_per_segment,
                end_time=segment_duration,
                request_rate=request_rate,
                pt_distributions_filename=pt_distributions_file
            )
            
            # Adjust timestamps to account for previous segments
            trace_df['arrival_timestamp'] += current_time_offset
            
            # Add columns to identify the rate and cycle
            # trace_df['request_rate'] = request_rate
            # trace_df['cycle'] = cycle + 1
            
            # Add to our list
            all_dfs.append(trace_df)
            
            # Update time offset for next segment
            current_time_offset += segment_duration
            
        # Second half of cycle: descending rates (150 to 10)
        for i, request_rate in enumerate(descending_rates):
            # print(f"  Generating segment with rate {request_rate} for {segment_duration}s")
            
            # Generate trace for this request rate and segment
            trace_df = generate_trace_from_prompt_token_size_distributions(
                max_requests=max_requests_per_segment,
                end_time=segment_duration,
                request_rate=request_rate,
                pt_distributions_filename=pt_distributions_file
            )
            
            # Adjust timestamps to account for previous segments
            trace_df['arrival_timestamp'] += current_time_offset
            
            # Add columns to identify the rate and cycle
            # trace_df['request_rate'] = request_rate
            # trace_df['cycle'] = cycle + 1
            
            # Add to our list
            all_dfs.append(trace_df)
            
            # Update time offset for next segment
            current_time_offset += segment_duration
    
    # Concatenate all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Sort by arrival timestamp
    combined_df = combined_df.sort_values('arrival_timestamp').reset_index(drop=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Combined trace saved to {output_file} with {len(combined_df)} total requests")
    print(f"Total simulation time: {current_time_offset} seconds")
    
    return combined_df

if __name__ == "__main__":
    np.random.seed(0)
    
    # Create traces directory if it doesn't exist
    traces_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'traces')
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)
    
    # Download Azure LLM traces if needed
    # download_azure_llm_traces()
    
    # Generate combined trace with changing request rates
    # 900s per cycle, 60s per segment, 10 cycles
    segment_duration = 30  # seconds per rate
    max_requests_per_segment = 80000  # Limit requests per segment
    cycles = 50  # Number of cycles
    
    # Generate for code distributions
    code_combined_trace = generate_long_rps_trace(
        max_requests_per_segment=max_requests_per_segment,
        segment_duration=segment_duration,
        cycles=cycles,
        pt_distributions_file="data/code_distributions.csv",
        output_file=os.path.join(traces_dir, "long_rps_code_combined.csv")
    )
    
    # Generate for conv distributions
    # conv_combined_trace = generate_long_rps_trace(
    #     max_requests_per_segment=max_requests_per_segment,
    #     segment_duration=segment_duration,
    #     cycles=cycles,
    #     pt_distributions_file="data/conv_distributions.csv",
    #     output_file=os.path.join(traces_dir, "long_rps_conv_combined.csv")
    # )
    
    print("All combined traces generated successfully!")