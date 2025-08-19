#!/usr/bin/env python3
"""Queue Size Box and Whisker Plot Generator

This script creates a box and whisker plot comparing server buffer accumulation
status across different numbers of connected clients.

Each CSV file represents a different number of clients:
- queue_size1.csv: 1 client
- queue_size4.csv: 4 clients  
- queue_size8.csv: 8 clients
- queue_size16.csv: 16 clients
- queue_size32.csv: 32 clients
- queue_size64.csv: 64 clients

The resulting PNG file is saved to the `paper_plot` directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------------
CSV_MAPPING: Dict[str, str] = {
    "packet_log/exp_1/queue_size1.csv": "1 Client",
    "packet_log/exp_1/queue_size4.csv": "4 Clients", 
    "packet_log/exp_1/queue_size8.csv": "8 Clients",
    "packet_log/exp_1/queue_size16.csv": "16 Clients",
    "packet_log/exp_1/queue_size32.csv": "32 Clients",
    "packet_log/exp_1/queue_size64.csv": "64 Clients",
}

OUTPUT_DIR = Path("paper_plot")
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------------------
# DATA LOADING
# --------------------------------------------------------------------------------------

def load_queue_data() -> Dict[str, pd.Series]:
    """Load queue size data from CSV files."""
    data: Dict[str, pd.Series] = {}
    
    for path, label in CSV_MAPPING.items():
        try:
            df = pd.read_csv(path)
            if "queue_size" not in df.columns:
                raise ValueError(f"Column 'queue_size' missing in {path}")
            
            data[label] = df["queue_size"]
            print(f"Loaded {label}: {len(df)} data points, queue_size range: {df['queue_size'].min()}-{df['queue_size'].max()}")
            
        except FileNotFoundError:
            print(f"Warning: {path} not found. Skipping {label}.")
        except Exception as exc:
            print(f"Error loading {path}: {exc}")
    
    return data

# --------------------------------------------------------------------------------------
# PLOTTING
# --------------------------------------------------------------------------------------

def create_box_whisker_plot(queue_data: Dict[str, pd.Series]) -> None:
    """Create box and whisker plot of queue sizes across different client counts."""
    # Set style for better visualization
    plt.style.use('seaborn-v0_8')
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for box plot
    labels = list(queue_data.keys())
    data_series = list(queue_data.values())
    
    # Create box plot
    box_plot = ax.boxplot(data_series, labels=labels, patch_artist=True)
    
    # Customize box plot colors
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightsteelblue', 'lightpink']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    
    # Customize median lines
    for median in box_plot['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    # Customize whiskers and caps
    for whisker in box_plot['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1)
    
    for cap in box_plot['caps']:
        cap.set_color('black')
        cap.set_linewidth(1)
    
    # Customize fliers (outliers)
    for flier in box_plot['fliers']:
        flier.set_marker('o')
        flier.set_markerfacecolor('red')
        flier.set_markeredgecolor('black')
        flier.set_markersize(4)
        flier.set_alpha(0.7)
    
    # Customize the plot
    ax.set_title('Server Buffer Accumulation Status\n(Queue Size Distribution by Number of Clients)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Connected Clients', fontsize=16, fontweight='bold')
    ax.set_ylabel('Queue Size', fontsize=16, fontweight='bold')
    
    # Set font sizes for ticks
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.4, linestyle='-', linewidth=0.5)
    
    # Add statistics annotations
    for i, (label, data) in enumerate(queue_data.items()):
        mean_val = data.mean()
        median_val = data.median()
        max_val = data.max()
        
        # Add mean line annotation
        ax.text(i+1, mean_val, f'μ={mean_val:.1f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add max value annotation if it's significantly different from mean
        if max_val > mean_val + 2 * data.std():
            ax.text(i+1, max_val, f'max={max_val}', 
                    ha='center', va='bottom', fontsize=9, color='red',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot with white background for better readability
    plt.savefig(OUTPUT_DIR / "queue_boxplot.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', transparent=False)
    plt.close()

def print_queue_statistics(queue_data: Dict[str, pd.Series]) -> None:
    """Print detailed statistics for each client count."""
    print("\nQueue Size Statistics:")
    print("=" * 80)
    
    for label, data in queue_data.items():
        print(f"\n{label}:")
        print(f"  Count: {len(data):>6}")
        print(f"  Mean:  {data.mean():>8.2f}")
        print(f"  Median:{data.median():>8.2f}")
        print(f"  Std:   {data.std():>8.2f}")
        print(f"  Min:   {data.min():>8.0f}")
        print(f"  Max:   {data.max():>8.0f}")
        print(f"  Q1:    {data.quantile(0.25):>8.2f}")
        print(f"  Q3:    {data.quantile(0.75):>8.2f}")

# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------

def main() -> None:
    """Main function to create and save the queue size box plot."""
    print("Loading queue size data from CSV files...")
    queue_data = load_queue_data()
    
    if not queue_data:
        print("No queue data loaded – aborting plot.")
        return
    
    print(f"\nCreating box and whisker plot for {len(queue_data)} client configurations...")
    create_box_whisker_plot(queue_data)
    
    print_queue_statistics(queue_data)
    
    print(f"\nQueue size box plot saved to: {OUTPUT_DIR / 'queue_boxplot.png'}")


if __name__ == "__main__":
    main() 