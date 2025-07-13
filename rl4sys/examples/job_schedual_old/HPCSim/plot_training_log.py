#!/usr/bin/env python3
"""
Script to plot training log data from CSV file.
Plots reward, sjf_score, and f1_score vs episode in three subplots.
Also includes a focused reward analysis with two subplots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np

def plot_training_log(csv_path, output_path=None):
    """
    Plot training log data from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        output_path (str, optional): Path to save the plot image
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check if required columns exist
    required_columns = ['episode', 'reward', 'sjf_score', 'f1_score', 'iteration']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with three subplots for main analysis
    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 10))
    fig1.suptitle('Training Log Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Reward vs Episode
    axes1[0].plot(df['episode'], df['reward'], linewidth=1, alpha=0.8, color='#1f77b4')
    axes1[0].set_title('Reward vs Episode', fontsize=14, fontweight='bold')
    axes1[0].set_ylabel('Reward', fontsize=12)
    axes1[0].grid(True, alpha=0.3)
    axes1[0].tick_params(axis='both', which='major', labelsize=10)
    
    # Plot 2: SJF Score vs Episode
    axes1[1].plot(df['episode'], df['sjf_score'], linewidth=1, alpha=0.8, color='#ff7f0e')
    axes1[1].set_title('SJF Score vs Episode', fontsize=14, fontweight='bold')
    axes1[1].set_ylabel('SJF Score', fontsize=12)
    axes1[1].grid(True, alpha=0.3)
    axes1[1].tick_params(axis='both', which='major', labelsize=10)
    
    # Plot 3: F1 Score vs Episode
    axes1[2].plot(df['episode'], df['f1_score'], linewidth=1, alpha=0.8, color='#2ca02c')
    axes1[2].set_title('F1 Score vs Episode', fontsize=14, fontweight='bold')
    axes1[2].set_xlabel('Episode', fontsize=12)
    axes1[2].set_ylabel('F1 Score', fontsize=12)
    axes1[2].grid(True, alpha=0.3)
    axes1[2].tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Create second figure for focused reward analysis
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
    fig2.suptitle('Focused Reward Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Reward vs Episode (all data)
    axes2[0].plot(df['episode'], df['reward'], linewidth=1, alpha=0.8, color='#d62728')
    axes2[0].set_title('Reward vs Episode', fontsize=14, fontweight='bold')
    axes2[0].set_ylabel('Reward', fontsize=12)
    axes2[0].grid(True, alpha=0.3)
    axes2[0].tick_params(axis='both', which='major', labelsize=10)
    
    # Calculate average reward per iteration (all data)
    # Group by iteration and calculate mean reward
    avg_reward_by_iteration = df.groupby('iteration')['reward'].mean().reset_index()
    
    # Plot 2: Average Reward vs Iteration
    axes2[1].plot(avg_reward_by_iteration['iteration'], avg_reward_by_iteration['reward'], 
                 linewidth=2, alpha=0.8, color='#9467bd', marker='o', markersize=4)
    axes2[1].set_title('Average Reward vs Iteration', fontsize=14, fontweight='bold')
    axes2[1].set_xlabel('Iteration', fontsize=12)
    axes2[1].set_ylabel('Average Reward', fontsize=12)
    axes2[1].grid(True, alpha=0.3)
    axes2[1].tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plots if output path is provided
    if output_path:
        # Save first plot
        base_name = output_path.rsplit('.', 1)[0]
        ext = output_path.rsplit('.', 1)[1] if '.' in output_path else 'png'
        
        plot1_path = f"{base_name}_main.{ext}"
        plot2_path = f"{base_name}_reward_focus.{ext}"
        
        fig1.savefig(plot1_path, dpi=300, bbox_inches='tight')
        fig2.savefig(plot2_path, dpi=300, bbox_inches='tight')
        print(f"Main plot saved to {plot1_path}")
        print(f"Reward focus plot saved to {plot2_path}")
    
    # Show the plots
    plt.show()

def main():
    """Main function to handle command line arguments and execute plotting."""
    parser = argparse.ArgumentParser(description='Plot training log data from CSV file')
    parser.add_argument('--csv_path', default='./examples/rl4sys_server/20250713_064230/episode_completion_data.csv', help='Path to the CSV file containing training log data')
    parser.add_argument('-o', '--output', default='./examples/rl4sys_server/20250713_064230/episode_completion_data.png', help='Output path for the plot image (optional)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: File {args.csv_path} does not exist")
        return
    
    # Plot the data
    plot_training_log(args.csv_path, args.output)

if __name__ == "__main__":
    main() 