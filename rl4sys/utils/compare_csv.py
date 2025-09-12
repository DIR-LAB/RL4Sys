import argparse
import os
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with standardized column names when possible.

    Expected CSV headers typically include "Wall time,Step,Value".
    This function attempts to map common variants onto these names.
    """
    column_map = {}
    for col in df.columns:
        lower = str(col).strip().lower().replace(" ", "_")

        if lower in {"wall_time", "walltime", "time", "timestamp"}:
            column_map[col] = "Wall time"
        elif lower in {"step", "global_step", "steps"}:
            column_map[col] = "Step"
        elif lower in {"value", "val", "metric", "reward"}:
            column_map[col] = "Value"
        else:
            # Keep original for unknown columns
            column_map[col] = col

    df = df.rename(columns=column_map)
    return df


def _prepare_series(
    df: pd.DataFrame,
    x_axis: Literal["Step", "Wall time"],
    y_axis: str = "Value",
    rolling: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract and clean x/y arrays from a dataframe.

    - Ensures numeric types
    - Sorts by x
    - Collapses duplicate x values by mean
    - Optionally applies a centered rolling mean to y
    """
    if x_axis not in df.columns:
        raise ValueError(f"x-axis column '{x_axis}' not found in CSV. Columns: {list(df.columns)}")
    if y_axis not in df.columns:
        raise ValueError(f"y-axis column '{y_axis}' not found in CSV. Columns: {list(df.columns)}")

    # Coerce numeric and drop rows with NaNs in required columns
    work = df[[x_axis, y_axis]].copy()
    work[x_axis] = pd.to_numeric(work[x_axis], errors="coerce")
    work[y_axis] = pd.to_numeric(work[y_axis], errors="coerce")
    work = work.dropna(subset=[x_axis, y_axis])

    # Sort by x and aggregate duplicates by mean
    work = work.sort_values(x_axis)
    work = work.groupby(x_axis, as_index=False, sort=True).mean(numeric_only=True)

    if rolling and rolling > 1:
        # Centered rolling mean on y; preserve x
        work[y_axis] = work[y_axis].rolling(window=rolling, center=True, min_periods=max(1, rolling // 2)).mean()
        work = work.dropna(subset=[y_axis])

    x = work[x_axis].to_numpy()
    y = work[y_axis].to_numpy()
    return x, y


def _align_on_union_index(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align two series on the union of their x-values via linear interpolation.

    Returns (x_common, y1_interp, y2_interp).
    """
    s1 = pd.Series(y1, index=x1)
    s2 = pd.Series(y2, index=x2)

    # Ensure strictly increasing indices to avoid interpolation issues
    s1 = s1[~s1.index.duplicated(keep="last")].sort_index()
    s2 = s2[~s2.index.duplicated(keep="last")].sort_index()

    x_common = np.union1d(s1.index.values, s2.index.values)

    s1a = s1.reindex(x_common).interpolate(method="index", limit_direction="both")
    s2a = s2.reindex(x_common).interpolate(method="index", limit_direction="both")

    return x_common, s1a.values.astype(float), s2a.values.astype(float)


def compare_and_plot(
    csv1: str,
    csv2: str,
    xaxis: Literal["step", "time"],
    label1: str | None = None,
    label2: str | None = None,
    title: str | None = None,
    show_diff: bool = True,
    rolling: int | None = None,
    output_path: str | None = None,
    dpi: int = 150,
) -> None:
    """Load two CSVs, align on the chosen x-axis, and generate comparison plots.

    - xaxis: 'step' uses the 'Step' column; 'time' uses 'Wall time'.
    - show_diff: add a second subplot with (series2 - series1).
    - rolling: optional centered rolling window size for smoothing values.
    - output_path: if provided, save the figure there; otherwise call plt.show().
    """
    if not os.path.isfile(csv1):
        raise FileNotFoundError(f"File not found: {csv1}")
    if not os.path.isfile(csv2):
        raise FileNotFoundError(f"File not found: {csv2}")

    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    df1 = _canonicalize_columns(df1)
    df2 = _canonicalize_columns(df2)

    x_col = "Step" if xaxis.lower() == "step" else "Wall time"

    x1, y1 = _prepare_series(df1, x_col, "Value", rolling=rolling)
    x2, y2 = _prepare_series(df2, x_col, "Value", rolling=rolling)

    x_common, y1a, y2a = _align_on_union_index(x1, y1, x2, y2)

    if label1 is None:
        label1 = os.path.splitext(os.path.basename(csv1))[0]
    if label2 is None:
        label2 = os.path.splitext(os.path.basename(csv2))[0]

    if show_diff:
        fig, (ax_main, ax_diff) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(11, 5))
        ax_diff = None

    # Main overlay plot
    ax_main.plot(x_common, y1a, label=label1, linewidth=1.5)
    ax_main.plot(x_common, y2a, label=label2, linewidth=1.5)
    ax_main.set_ylabel("Value")
    ax_main.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax_main.legend()

    x_label = "Step" if x_col == "Step" else "Wall time"
    if title is None:
        title = f"Comparison on {x_label}"
    fig.suptitle(title)

    # Difference subplot
    if ax_diff is not None:
        diff = y2a - y1a
        ax_diff.plot(x_common, diff, color="tab:gray", linewidth=1.2)
        ax_diff.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
        ax_diff.set_ylabel("Δ (2-1)")
        ax_diff.set_xlabel(x_label)
        ax_diff.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    else:
        ax_main.set_xlabel(x_label)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two CSV files with columns 'Wall time,Step,Value'. "
            "Align by x-axis (Step or Wall time), overlay Value curves, and optionally plot differences."
        )
    )
    parser.add_argument("--csv1", default="/Users/girigiri_yomi/Udel_Proj/RL4Sys/rl4sys/examples/dgap/csv_data/1756992638__0.csv", type=str, help="Path to first CSV file")
    parser.add_argument("--csv2", default="/Users/girigiri_yomi/Udel_Proj/RL4Sys/rl4sys/examples/dgap/csv_data/1757103208.csv", type=str, help="Path to second CSV file")
    parser.add_argument(
        "--xaxis",
        type=str,
        choices=["step", "time"],
        default="step",
        help="Choose x-axis: 'step' (Step) or 'time' (Wall time)",
    )
    parser.add_argument("--label1", default="rl4sys", type=str, help="Legend label for first CSV")
    parser.add_argument("--label2", default="noml", type=str, help="Legend label for second CSV")
    parser.add_argument("--title", default="Comparison of RL4Sys and Noml", type=str, help="Figure title")
    parser.add_argument("--no-diff", dest="show_diff", action="store_false", help="Disable difference subplot")
    parser.set_defaults(show_diff=True)
    parser.add_argument("--rolling", type=int, default=None, help="Centered rolling window size for smoothing values")
    parser.add_argument("--output", type=str, default=None, help="Output image path (e.g., compare.png). If omitted, shows window")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    compare_and_plot(
        csv1=args.csv1,
        csv2=args.csv2,
        xaxis=args.xaxis,
        label1=args.label1,
        label2=args.label2,
        title=args.title,
        show_diff=args.show_diff,
        rolling=args.rolling,
        output_path=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()


