#!/usr/bin/env python3
"""
Performance Comparison Bar Plot Generator

This script creates a bar plot comparing the steps per second performance 
across different runtime conditions based on the provided table data.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Any
import seaborn as sns


def create_performance_data() -> Dict[str, float]:
    """
    Create a dictionary containing the performance data from the table.
    
    Returns:
        Dict[str, float]: Dictionary with condition names as keys and steps/s mean as values.
    """
    data = {
        "Random": 2490.2,
        "NN Infer": 1976.7,
        "NN Inf+Traj\n(SF=10)": 1863.7,
        "NN Inf+Traj\n(SF=5)": 1853.7,
        "NN Inf+Traj\n(SF=3)": 1865.8,
        "NN Inf+Traj\n(SF=1)": 1860.6,
        "Rllib local": 615.2,
        "Rllib remote": 232.6
    }
    return data


def create_color_mapping() -> Dict[str, str]:
    """
    Create a color mapping for different condition types based on the table formatting.
    
    Returns:
        Dict[str, str]: Dictionary mapping condition names to colors.
    """
    color_mapping = {
        # Light blue background conditions
        "Random": "lightblue",
        "NN Infer": "lightblue",
        # Light green background conditions
        "NN Inf+Traj\n(SF=10)": "lightgreen",
        "NN Inf+Traj\n(SF=5)": "lightgreen",
        "NN Inf+Traj\n(SF=3)": "lightgreen",
        "NN Inf+Traj\n(SF=1)": "lightgreen",
        # Light yellow background conditions
        "Rllib local": "lightyellow",
        "Rllib remote": "lightyellow"
    }
    return color_mapping


def create_performance_bar_plot(data: Dict[str, float], 
                               color_mapping: Dict[str, str], 
                               figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Create a bar plot comparing performance across different conditions.
    
    Args:
        data: Dictionary with condition names as keys and steps/s mean as values.
        color_mapping: Dictionary mapping condition names to colors.
        figsize: Tuple specifying the figure size (width, height).
    
    Returns:
        plt.Figure: The created matplotlib figure.
    """
    # Extract data for plotting
    conditions = list(data.keys())
    performance_values = list(data.values())
    colors = [color_mapping[condition] for condition in conditions]
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the bar plot with original colors
    bars = ax.bar(conditions, performance_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize the plot
    ax.set_title('End to End Runtime Performance Comparison\n(Steps per Second)', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Condition', fontsize=16, fontweight='bold')
    ax.set_ylabel('Steps per Second (mean)', fontsize=16, fontweight='bold')
    
    # Set x-axis labels without rotation and with larger font
    plt.xticks(fontsize=14)
    
    # Set y-axis label font size
    plt.yticks(fontsize=12)
    
    # Add value labels on top of each bar with larger font
    for bar, value in zip(bars, performance_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # Add horizontal grid lines on each y scale
    ax.grid(True, axis='y', alpha=0.5, linestyle='-', linewidth=0.5)
    
    # Adjust layout to prevent label cutoff and accommodate multi-line labels
    plt.tight_layout()
    
    return fig


def main() -> None:
    """
    Main function to create and display the performance plots.
    """
    import os
    
    # Set style for better visualization
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Configure matplotlib for better font rendering
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Create the data and color mapping
    performance_data = create_performance_data()
    color_mapping = create_color_mapping()
    
    # Create the main bar plot
    print("Creating main performance bar plot...")
    fig = create_performance_bar_plot(performance_data, color_mapping)
    
    # Create paper_plot directory if it doesn't exist
    os.makedirs("paper_plot", exist_ok=True)
    
    # Save the plot as PNG with transparent background
    output_path = os.path.join("paper_plot", "performance.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='none', edgecolor='none', transparent=True)
    print(f"Plot saved to: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print("=" * 50)
    for condition, value in performance_data.items():
        print(f"{condition:<20}: {value:>8.1f} steps/s")
    
    print(f"\nBest performer: {max(performance_data, key=performance_data.get)} ({max(performance_data.values()):.1f} steps/s)")
    print(f"Worst performer: {min(performance_data, key=performance_data.get)} ({min(performance_data.values()):.1f} steps/s)")


if __name__ == "__main__":
    main() 