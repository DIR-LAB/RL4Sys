from __future__ import annotations

"""Utility to plot reward curves for Lustre experiments.

This script reads two CSV files generated during reinforcement learning
experiments (baseline and RL-enhanced runs) and plots their reward curves on
one figure. The resulting image is saved to ``paper_plot/lustre.png`` with a
transparent background so that it can be embedded in documents without a
white rectangle surrounding it.

Usage
-----
Simply execute the script from the project root::

    python plot_lustre.py

The script assumes that the CSV files are located at::

    paper_plot/rl reward/lustre_base.csv
    paper_plot/rl reward/lustre_rl.csv

and that the output directory ``paper_plot`` already exists (it does not
attempt to create it).
"""

from pathlib import Path
from typing import Final

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Paths to the CSV files containing the reward curves.
CSV_BASE: Final[Path] = Path("paper_plot/rl reward/lustre_base.csv")
CSV_RL: Final[Path] = Path("paper_plot/rl reward/lustre_rl.csv")

# Output path for the generated plot.
OUTPUT_PNG: Final[Path] = Path("paper_plot/lustre.png")

# Plot appearance settings.
FIG_SIZE: Final[tuple[float, float]] = (10, 6)
LINE_WIDTH: Final[float] = 1.5

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file produced by TensorBoard and return it as a DataFrame.

    The CSVs are expected to have at least the columns ``Step`` and ``Value``.

    Parameters
    ----------
    path: Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the CSV data.
    """
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    expected_columns: set[str] = {"Step", "Value"}
    if not expected_columns.issubset(df.columns):
        missing = expected_columns - set(df.columns)
        raise ValueError(f"Missing expected columns {missing} in {path}")

    return df


# ---------------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------------

def plot_reward_curves() -> None:
    """Plot reward curves from the baseline and RL CSV files.

    The function loads the CSVs, plots ``Step`` vs ``Value`` for each, and
    saves the resulting figure to :pydata:`OUTPUT_PNG` with a transparent
    background.
    """
    # Load data
    df_base = _load_csv(CSV_BASE)
    df_rl = _load_csv(CSV_RL)

    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=150, facecolor="none")
    fig.patch.set_alpha(0.0)  # Fully transparent figure background
    ax.set_facecolor("none")  # Transparent axes background

    # Plot curves
    ax.plot(
        df_base["Step"],
        df_base["Value"],
        label="Baseline",
        linewidth=LINE_WIDTH,
        color="tab:blue",
    )
    ax.plot(
        df_rl["Step"],
        df_rl["Value"],
        label="RL4Sys",
        linewidth=LINE_WIDTH,
        color="tab:orange",
    )

    # Labels and legend
    ax.set_title("Lustre Reward Curves")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (Value)")
    ax.legend()

    # Improve layout and save
    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, transparent=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_reward_curves()
