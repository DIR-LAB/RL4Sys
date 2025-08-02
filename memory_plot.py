#!/usr/bin/env python3
"""
Memory Usage Plot Generator

This script creates a line plot comparing memory usage (RSS) across different 
experimental conditions with normalized timestamps.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import seaborn as sns
import os
from datetime import datetime


def load_memory_data() -> Dict[str, pd.DataFrame]:
    """
    Load memory data from CSV files and normalize timestamps.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping condition names to dataframes.
    """
    # Define file paths and corresponding condition names
    file_mapping = {
        "memtest/exp_1 copy/job_main_rand_memory_profile_20250728_171221.csv": "Random",
        "memtest/exp_1 copy/job_main_infer_memory_profile_20250728_171307.csv": "NN Infer", 
        "memtest/exp_1 copy/job_main_memory_profile_20250728_170410 10 SF.csv": "RL4Sys SF=10",
        "memtest/exp_1 copy/job_main_memory_profile_20250728_170502_5SF.csv": "RL4Sys SF=5",
        "memtest/exp_1 copy/job_main_memory_profile_20250728_170608_3SF.csv": "RL4Sys SF=3",
        "memtest/exp_1 copy/job_main_memory_profile_20250728_170647_1SF.csv": "RL4Sys SF=1",
        "memtest/exp_1 copy/rllib_client_job_remote_memory_profile_20250728_171607.csv": "RLlib Remote",
        "memtest/exp_1 copy/rllib_client_job_local_memory_profile_20250728_171436 copy.csv": "RLlib Local"
    }
    
    data_dict = {}
    
    for file_path, condition_name in file_mapping.items():
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['process_timestamp'])
            
            # Normalize timestamp to start from 0
            start_time = df['timestamp'].min()
            df['normalized_time'] = (df['timestamp'] - start_time).dt.total_seconds()
            
            # Convert RSS from bytes to MB for better readability
            df['rss_mb'] = df['process_rss'] / (1024 * 1024)
            
            data_dict[condition_name] = df
            
            print(f"Loaded {condition_name}: {len(df)} data points")
            
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found, skipping {condition_name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return data_dict


def create_memory_plot(data_dict: Dict[str, pd.DataFrame], 
                      figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create a line plot comparing memory usage across different conditions.
    
    Args:
        data_dict: Dictionary mapping condition names to dataframes.
        figsize: Tuple specifying the figure size (width, height).
    
    Returns:
        plt.Figure: The created matplotlib figure.
    """
    # Set style for better visualization
    plt.style.use('seaborn-v0_8')
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors for different conditions with better contrast
    colors = {
        "Random": "#1f77b4",      # Blue
        "NN Infer": "#ff7f0e",    # Orange
        "RL4Sys SF=10": "#2ca02c", # Green
        "RL4Sys SF=5": "#d62728",  # Red
        "RL4Sys SF=3": "#9467bd",  # Purple
        "RL4Sys SF=1": "#8c564b",  # Brown
        "RLlib Remote": "#e377c2", # Pink
        "RLlib Local": "#7f7f7f"   # Gray
    }
    
    # Plot each condition
    for condition_name, df in data_dict.items():
        if len(df) > 0:
            color = colors.get(condition_name, "gray")
            ax.plot(df['normalized_time'], df['rss_mb'], 
                   label=condition_name, color=color, linewidth=3, alpha=0.9)
    
    # Customize the plot
    ax.set_title('Memory Usage Comparison\n(RSS over Time)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Time (seconds)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontsize=16, fontweight='bold')
    
    # Set font sizes for ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid with better visibility
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    
    # Add legend with better positioning
    ax.legend(fontsize=14, loc='upper left', framealpha=0.95, 
             fancybox=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_detailed_memory_plot(data_dict: Dict[str, pd.DataFrame], 
                               figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Create a detailed memory plot with subplots for each condition.
    
    Args:
        data_dict: Dictionary mapping condition names to dataframes.
        figsize: Tuple specifying the figure size (width, height).
    
    Returns:
        plt.Figure: The created matplotlib figure.
    """
    # Set style for better visualization
    plt.style.use('seaborn-v0_8')
    
    # Create subplots
    n_conditions = len(data_dict)
    fig, axes = plt.subplots(n_conditions, 1, figsize=figsize, sharex=True)
    
    # If only one condition, make axes iterable
    if n_conditions == 1:
        axes = [axes]
    
    # Define colors with better contrast
    colors = {
        "Random": "#1f77b4",      # Blue
        "NN Infer": "#ff7f0e",    # Orange
        "RL4Sys SF=10": "#2ca02c", # Green
        "RL4Sys SF=5": "#d62728",  # Red
        "RL4Sys SF=3": "#9467bd",  # Purple
        "RL4Sys SF=1": "#8c564b",  # Brown
        "RLlib Remote": "#e377c2", # Pink
        "RLlib Local": "#7f7f7f"   # Gray
    }
    
    # Plot each condition in its own subplot
    for i, (condition_name, df) in enumerate(data_dict.items()):
        if len(df) > 0:
            color = colors.get(condition_name, "gray")
            axes[i].plot(df['normalized_time'], df['rss_mb'], 
                        color=color, linewidth=3, alpha=0.9)
            
            axes[i].set_title(f'{condition_name}', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Memory (MB)', fontsize=12)
            axes[i].grid(True, alpha=0.4, linewidth=0.8)
            axes[i].tick_params(axis='both', which='major', labelsize=10)
    
    # Set x-axis label for the bottom subplot
    axes[-1].set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    
    # Add overall title
    fig.suptitle('Memory Usage by Condition', fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def print_memory_statistics(data_dict: Dict[str, pd.DataFrame]) -> None:
    """
    Print memory usage statistics for each condition.
    
    Args:
        data_dict: Dictionary mapping condition names to dataframes.
    """
    print("\nMemory Usage Statistics:")
    print("=" * 60)
    
    for condition_name, df in data_dict.items():
        if len(df) > 0:
            rss_mb = df['rss_mb']
            print(f"{condition_name:<15}:")
            print(f"  Mean: {rss_mb.mean():>8.1f} MB")
            print(f"  Max:  {rss_mb.max():>8.1f} MB")
            print(f"  Min:  {rss_mb.min():>8.1f} MB")
            print(f"  Std:  {rss_mb.std():>8.1f} MB")
            print(f"  Duration: {df['normalized_time'].max():>8.1f} seconds")
            print()


def main() -> None:
    """
    Main function to create and save the memory usage plots.
    """
    # Load memory data
    print("Loading memory data from CSV files...")
    data_dict = load_memory_data()
    
    if not data_dict:
        print("No data loaded. Please check file paths.")
        return
    
    # Create the main memory plot
    print("Creating memory usage plot...")
    fig = create_memory_plot(data_dict)
    
    # Create paper_plot directory if it doesn't exist
    os.makedirs("paper_plot", exist_ok=True)
    
    # Save the plot as PNG with white background for better readability
    output_path = os.path.join("paper_plot", "memory.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', transparent=False)
    print(f"Memory plot saved to: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Create detailed plot
    print("Creating detailed memory plot...")
    detailed_fig = create_detailed_memory_plot(data_dict)
    
    # Save the detailed plot with white background
    detailed_output_path = os.path.join("paper_plot", "memory_detailed.png")
    detailed_fig.savefig(detailed_output_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none', transparent=False)
    print(f"Detailed memory plot saved to: {detailed_output_path}")
    
    # Close the detailed figure
    plt.close(detailed_fig)
    
    # Print statistics
    print_memory_statistics(data_dict)


if __name__ == "__main__":
    main() 