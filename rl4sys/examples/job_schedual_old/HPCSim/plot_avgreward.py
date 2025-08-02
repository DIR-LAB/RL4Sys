import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_avg_reward(csv_file: str) -> None:
    """Plot average reward per iteration from a CSV log file.
    
    Reads the CSV file containing training data, groups episodes by iteration,
    calculates average reward for each iteration, and creates a plot.
    
    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing training logs.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Group by iteration and calculate average reward
    # The CSV already has iteration and episode columns
    avg_rewards = df.groupby('iteration')['reward'].mean().reset_index()
    avg_rewards.columns = ['epoch', 'reward']
    
    # Filter data - use first 100 iterations after calculating averages
    avg_rewards = avg_rewards.head(100)
    print(f"Using first 100 iterations for {Path(csv_file).stem}")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the plot
    ax.plot(avg_rewards['epoch'], avg_rewards['reward'], 
            linewidth=2.5, alpha=0.8, color='#2E86AB', marker='o', markersize=6)
    
    # Add a moving average line for smoother trend visualization
    window_size = min(5, len(avg_rewards) // 4)  # Adaptive window size
    if len(avg_rewards) > window_size and window_size > 1:
        moving_avg = avg_rewards['reward'].rolling(window=window_size, center=True).mean()
        ax.plot(avg_rewards['epoch'], moving_avg, 
                linewidth=3, alpha=0.9, color='#A23B72')
    
    # Customize the plot
    ax.set_xlabel('Epoch', fontsize=20, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=20, fontweight='bold')
    
    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout and save
    plt.tight_layout()
    
    # Save the plot in the same directory as the CSV file
    output_path = Path(csv_file).parent / f"{Path(csv_file).stem}_avg_reward.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    print(f"Average reward plot saved to: {output_path}")
    print(f"Total epochs: {len(avg_rewards)}")
    print(f"Final average reward: {avg_rewards['reward'].iloc[-1]:.2f}")

def main():
    """Main function to plot average rewards for both CSV files."""
    # Define the two CSV file paths
    csv_file1 = "./log/20250730_043148/SDSC-SP2-1998-4.2-cln.csv"
    csv_file2 = "./log/20250730_043145/lublin_256.csv"
    
    # Plot for first CSV file
    print("=" * 60)
    print("Plotting average rewards for first CSV file...")
    print("=" * 60)
    plot_avg_reward(csv_file1)
    
    # Plot for second CSV file
    print("\n" + "=" * 60)
    print("Plotting average rewards for second CSV file...")
    print("=" * 60)
    plot_avg_reward(csv_file2)

if __name__ == "__main__":
    main()
