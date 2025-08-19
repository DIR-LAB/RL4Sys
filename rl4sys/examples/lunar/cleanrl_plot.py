import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_training_progress(csv_file: str = "./logs/lunar_clearl/LunarLander-v3__cleanrl_Lunar__1__1753769782.csv") -> None:
    """Plot the training progress from the CSV log file.
    
    Reads the CSV file containing training data and creates a plot showing
    the accumulated reward over training iterations.
    
    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing training logs.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the plot
    ax.plot(df['global_step'], df['accumulated_reward'], 
            linewidth=2, alpha=0.8, color='#2E86AB')
    
    # Add a moving average line for smoother trend visualization
    window_size = 20
    if len(df) > window_size:
        moving_avg = df['accumulated_reward'].rolling(window=window_size, center=True).mean()
        ax.plot(df['global_step'], moving_avg, 
                linewidth=3, alpha=0.9, color='#A23B72', 
                label=f'Moving Average (window={window_size})')
        ax.legend()
    
    # Customize the plot
    ax.set_xlabel('Global Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accumulated Reward', fontsize=12, fontweight='bold')
    ax.set_title('LunarLander-v3 Training Progress\nCleanRL PPO Implementation', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add some statistics as text
    final_reward = df['accumulated_reward'].iloc[-1]
    max_reward = df['accumulated_reward'].max()
    min_reward = df['accumulated_reward'].min()
    avg_reward = df['accumulated_reward'].mean()
    
    stats_text = f'Final Reward: {final_reward:.2f}\n'
    stats_text += f'Max Reward: {max_reward:.2f}\n'
    stats_text += f'Min Reward: {min_reward:.2f}\n'
    stats_text += f'Avg Reward: {avg_reward:.2f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8), fontsize=10)
    
    # Tight layout and save
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("./plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "lunar_cleanrl_training_progress.png", 
                dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    print(f"Training progress plot saved to: {output_dir / 'lunar_cleanrl_training_progress.png'}")
    print(f"Total training steps: {len(df)}")
    print(f"Final accumulated reward: {final_reward:.2f}")

if __name__ == "__main__":
    # Plot the training progress
    plot_training_progress()





