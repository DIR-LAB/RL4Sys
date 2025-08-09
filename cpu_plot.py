#!/usr/bin/env python3
"""CPU Usage Plot Generator

Generates two plots:
1. Bar plot comparing mean CPU usage per core across experimental conditions.
2. Time-series plot showing CPU usage per core over normalised time.

CSV files are expected in `cputest/exp_1/` and must contain a
`process_cpu_percent_per_core` column.
The resulting PNG files are saved to the `paper_plot` directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------
CSV_MAPPING: Dict[str, str] = {
    "cputest/exp_1/job_main_rand 100 traj send_cpu_profile_20250728_171221.csv": "Random",
    "cputest/exp_1/job_main_infer 100 traj send_cpu_profile_20250728_171307.csv": "Infer-Only",
    "cputest/exp_1/job_main 10 traj send_cpu_profile_20250728_170502_10SF.csv": "RL4Sys SF=10",
    "cputest/exp_1/job_main 10 traj send_cpu_profile_20250728_170543_5SF.csv": "RL4Sys SF=5",
    "cputest/exp_1/job_main 10 traj send_cpu_profile_20250728_170608_3SF.csv": "RL4Sys SF=3",
    "cputest/exp_1/job_main 10 traj send_cpu_profile_20250728_170647_1SF.csv": "RL4Sys SF=1",
    "cputest/exp_1/rllib_client_job_remote_cpu_profile_20250728_171607.csv": "RLlib Remote",
    "cputest/exp_1/rllib_client_job_local_cpu_profile_20250728_171436.csv": "RLlib Local",
}

COLOR_MAPPING: Dict[str, str] = {
    "Random": "lightblue",
    "Infer-Only": "lightblue",
    "RL4Sys SF=10": "lightgreen",
    "RL4Sys SF=5": "lightgreen",
    "RL4Sys SF=3": "lightgreen",
    "RL4Sys SF=1": "lightgreen",
    "RLlib Remote": "lightyellow",
    "RLlib Local": "lightyellow",
}

OUTPUT_DIR = Path("paper_plot")
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------------------

def load_cpu_data() -> Dict[str, pd.DataFrame]:
    """Load CPU csv files and add normalised time column."""
    data: Dict[str, pd.DataFrame] = {}
    for path, cond in CSV_MAPPING.items():
        try:
            df = pd.read_csv(path)
            if "process_cpu_percent_per_core" not in df.columns:
                raise ValueError(f"Column 'process_cpu_percent_per_core' missing in {path}")

            df["timestamp"] = pd.to_datetime(df["process_timestamp"])
            start = df["timestamp"].min()
            df["norm_time"] = (df["timestamp"] - start).dt.total_seconds()
            data[cond] = df
            print(f"Loaded {cond}: {len(df)} rows")
        except FileNotFoundError:
            print(f"Warning: {path} not found. Skipping {cond}.")
        except Exception as exc:
            print(f"Error loading {path}: {exc}")
    return data

# --------------------------------------------------------------------------------------
# PLOTTING UTILITIES
# --------------------------------------------------------------------------------------

def bar_plot(cpu_data: Dict[str, pd.DataFrame]) -> None:
    """Create bar plot of mean CPU-percent-per-core per condition."""
    means = {cond: df["process_cpu_percent_per_core"].mean() for cond, df in cpu_data.items() if not df.empty}

    conds = list(means.keys())
    values = [means[c] for c in conds]
    colors = [COLOR_MAPPING.get(c, "gray") for c in conds]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(conds, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_title('CPU Usage per Core Comparison\n(Average CPU Usage)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Condition', fontsize=16, fontweight='bold')
    ax.set_ylabel('CPU Usage (% per core)', fontsize=16, fontweight='bold')
    
    # Set x-axis labels without rotation and with larger font
    plt.xticks(fontsize=14)
    
    # Set y-axis label font size
    plt.yticks(fontsize=12)
    
    # Add value labels on top of each bar with larger font
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # Add horizontal grid lines on each y scale
    ax.grid(True, axis='y', alpha=0.5, linestyle='-', linewidth=0.5)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / "cpu_bar.png", dpi=300, bbox_inches='tight', 
                facecolor='none', edgecolor='none', transparent=True)
    plt.close()


def time_series_plot(cpu_data: Dict[str, pd.DataFrame]) -> None:
    """Create time-series plot of CPU usage per core over normalised time."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for cond, df in cpu_data.items():
        if df.empty:
            continue
        color = COLOR_MAPPING.get(cond, "gray")
        ax.plot(df["norm_time"], df["process_cpu_percent_per_core"], 
               label=cond, color=color, linewidth=2, alpha=0.8)

    ax.set_title('CPU Usage per Core Over Time (Normalised)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Time (s)', fontsize=16, fontweight='bold')
    ax.set_ylabel('CPU Usage (% per core)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / "cpu_time.png", dpi=300, bbox_inches='tight',
                facecolor='none', edgecolor='none', transparent=True)
    plt.close()

# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------

def main() -> None:
    # Set style for better visualization
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Configure matplotlib for better font rendering
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (12, 8)
    
    cpu_data = load_cpu_data()
    if not cpu_data:
        print("No CPU data loaded – aborting plots.")
        return

    bar_plot(cpu_data)
    time_series_plot(cpu_data)
    print("CPU plots saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main() 