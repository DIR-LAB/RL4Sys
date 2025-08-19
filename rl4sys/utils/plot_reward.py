import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_reward(csv_file: str) -> None:
    """Plot reward values from a CSV log file.
    
    Reads the CSV file containing training data and creates a plot showing
    reward values over training steps.
    
    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing training logs.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter data based on environment type
    filename = Path(csv_file).stem
    if 'lunar' in filename.lower() or 'Lunar' in filename:
        # For Lunarlander environments, use first 500 epochs
        df = df.head(500)
        print(f"Using first 500 epochs for Lunarlander environment")
    else:
        # For other environments (Lublin, SDSC), use first 100 episodes
        df = df.head(100)
        print(f"Using first 100 episodes for {filename}")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the plot
    ax.plot(df['Step'], df['Value'], 
            linewidth=2, alpha=0.8, color='#2E86AB', marker='o', markersize=4)
    
    # Add a moving average line for smoother trend visualization
    window_size = min(20, len(df) // 10)  # Adaptive window size
    if len(df) > window_size and window_size > 1:
        moving_avg = df['Value'].rolling(window=window_size, center=True).mean()
        ax.plot(df['Step'], moving_avg, 
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
    csv_path = Path(csv_file)
    output_path = csv_path.parent / f"{csv_path.stem}_reward_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    print(f"Reward plot saved to: {output_path}")
    print(f"Total epochs: {len(df)}")
    print(f"Final reward: {df['Value'].iloc[-1]:.2f}")

def main():
    """Main function to plot rewards for all four CSV files."""
    # Define the four CSV file paths
    csv_files = [
        "paper_plot/rl reward/cleanrl_lunar.csv",
        "paper_plot/rl reward/RL4Sys lublin 256.csv", 
        "paper_plot/rl reward/RL4Sys Lunar.csv",
        "paper_plot/rl reward/RL4Sys SDSC SP2.csv"
    ]
    
    # Plot for each CSV file
    for i, csv_file in enumerate(csv_files, 1):
        print("=" * 60)
        print(f"Plotting rewards for CSV file {i}: {csv_file}")
        print("=" * 60)
        try:
            plot_reward(csv_file)
        except Exception as e:
            print(f"Error plotting {csv_file}: {e}")
        print()

if __name__ == "__main__":
    main()
