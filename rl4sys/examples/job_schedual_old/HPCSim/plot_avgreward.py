import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _prepare_baseline_df(csv_file: str, max_iters: int = 100) -> pd.DataFrame:
    """Load baseline CSV and compute average reward per iteration.

    Parameters
    ----------
    csv_file: str
        Path to baseline CSV file containing columns ``iteration`` and ``reward``.
    max_iters: int, default 100
        Number of iterations to keep (starting from 0).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``epoch`` and ``reward`` limited to *max_iters*.
    """
    df = pd.read_csv(csv_file)
    avg_rewards = df.groupby("iteration")["reward"].mean().reset_index()
    avg_rewards.columns = ["epoch", "reward"]
    return avg_rewards.head(max_iters)


def _prepare_rl4sys_df(csv_file: str, max_iters: int = 100) -> pd.DataFrame:
    """Load RL4Sys reward CSV (Step/Value) and trim to first *max_iters* rows."""
    df = pd.read_csv(csv_file)
    df = df.rename(columns={"Step": "epoch", "Value": "reward"})
    return df.loc[: max_iters - 1, ["epoch", "reward"]]


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def plot_comparison(
    dataset_name: str,
    baseline_csv: str,
    rl4sys_csv: str,
    output_dir: Path | None = None,
    max_iters: int = 100,
) -> Path:
    """Plot baseline vs RL4Sys reward curves for a single dataset.

    Parameters
    ----------
    dataset_name: str
        Human-readable identifier used in the plot title.
    baseline_csv: str
        Path to the baseline training CSV.
    rl4sys_csv: str
        Path to the RL4Sys reward CSV produced by TensorBoard export.
    output_dir: pathlib.Path | None, optional
        Directory to save the resulting PNG. Defaults to the baseline CSV's
        parent directory.
    max_iters: int, default 100
        Number of iterations to plot.

    Returns
    -------
    pathlib.Path
        Path of the saved plot.
    """

    # Prepare dataframes
    df_base = _prepare_baseline_df(baseline_csv, max_iters=max_iters)
    df_rl = _prepare_rl4sys_df(rl4sys_csv, max_iters=max_iters)

    plt.style.use("seaborn-v0_8")
    # Transparent figure/axes
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="none")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    ax.plot(
        df_base["epoch"],
        df_base["reward"],
        label="Baseline (Avg per Iter)",
        linewidth=2.5,
        color="#1f77b4",
        marker="o",
        markersize=6,
    )

    ax.plot(
        df_rl["epoch"],
        df_rl["reward"],
        label="RL4Sys",
        linewidth=2.5,
        color="#ff7f0e",
        marker="s",
        markersize=6,
    )

    # Labels and title
    ax.set_title(f"Average Reward Comparison – {dataset_name}", fontsize=22, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=18, fontweight="bold")
    ax.set_ylabel("Reward", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14)

    fig.tight_layout()

    if output_dir is None:
        output_dir = Path("paper_plot")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name.lower().replace(' ', '_')}_reward_comparison.png"
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="none",
        edgecolor="none",
        transparent=True,
    )
    plt.close(fig)

    print(f"Saved comparison plot to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Legacy single-file plot kept for backward compatibility
# ---------------------------------------------------------------------------


def plot_avg_reward(csv_file: str) -> None:
    """Plot average reward per iteration from a CSV log file.
    
    Reads the CSV file containing training data, groups episodes by iteration,
    calculates average reward for each iteration, and creates a plot.
    
    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing training logs.
    """
    avg_rewards = _prepare_baseline_df(csv_file)
    
    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the plot
    ax.plot(
        avg_rewards["epoch"],
        avg_rewards["reward"],
        linewidth=2.5,
        alpha=0.8,
        color="#2E86AB",
        marker="o",
        markersize=6,
    )
    
    # Add a moving average line for smoother trend visualization
    window_size = min(5, len(avg_rewards) // 4)  # Adaptive window size
    if len(avg_rewards) > window_size and window_size > 1:
        moving_avg = avg_rewards['reward'].rolling(window=window_size, center=True).mean()
        ax.plot(avg_rewards['epoch'], moving_avg, 
                linewidth=3, alpha=0.9, color='#A23B72')
    
    # Customize the plot
    ax.set_xlabel("Epoch", fontsize=20, fontweight="bold")
    ax.set_ylabel("Reward", fontsize=20, fontweight="bold")
    
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

def main() -> None:
    """Generate comparison plots for SDSC and Lublin datasets."""

    # Baseline CSVs (already averaged per episode/iteration)
    baseline_sdsc = "./log/20250730_043148/SDSC-SP2-1998-4.2-cln.csv"
    baseline_lublin = "./log/20250730_043145/lublin_256.csv"

    # RL4Sys reward CSVs exported from TensorBoard scalars
    rl4sys_sdsc = "paper_plot/rl reward/RL4Sys SDSC SP2.csv"
    rl4sys_lublin = "paper_plot/rl reward/RL4Sys lublin 256.csv"

    print("Generating comparison plots...\n" + "-" * 60)

    plot_comparison("SDSC", baseline_sdsc, rl4sys_sdsc)
    plot_comparison("Lublin", baseline_lublin, rl4sys_lublin)

if __name__ == "__main__":
    main()
