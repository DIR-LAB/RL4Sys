from __future__ import annotations

"""Trajectory buffer memory logger.

This module provides a :class:`TrajectoryBufferLogger` that can be used to
periodically measure and persist the size (both in number of trajectories and
in memory footprint) of a trajectory buffer.

Logging is performed in CSV format inside a dedicated directory structure::

    memtrajtest/<timestamp>/<project_name>.csv

where *timestamp* is generated when the logger instance is created using the
format ``%Y%m%d_%H%M%S`` and *project_name* is provided by the user.

The class is intentionally lightweight and thread-safe so that it can be used
from asynchronous contexts (e.g. the trajectory sending thread in
:pyclass:`rl4sys.client.agent.RL4SysAgent`).
"""

from datetime import datetime
from pathlib import Path
import csv
import threading
from typing import Iterable, List

from pympler import asizeof

__all__ = ["TrajectoryBufferLogger"]


class TrajectoryBufferLogger:  # pylint: disable=too-few-public-methods
    """Utility class to record trajectory buffer statistics.

    Parameters
    ----------
    project_name:
        A human-readable identifier for the current experiment.  The final CSV
        file will be called ``<project_name>.csv``.
    base_dir:
        Base directory that will contain all runtime sub-folders.  A new
        sub-directory named with the current timestamp will be created inside
        *base_dir* on instantiation.  Defaults to ``"memtrajtest"``.

    Notes
    -----
    * The logger **never** deletes existing files or directories.
    * CSV columns are: ``timestamp,buffer_len,memory_bytes``.
    * Calling :py:meth:`log` from multiple threads is safe.
    """

    _CSV_HEADER: List[str] = ["timestamp", "buffer_len", "memory_bytes"]

    def __init__(self, project_name: str, base_dir: str | Path = "memtrajtest") -> None:
        self._project_name: str = project_name
        self._base_dir: Path = Path(base_dir)
        self._run_dir: Path = self._base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path: Path = self._run_dir / f"{self._project_name}.csv"
        self._lock = threading.Lock()

        # Write header only if the file is new/empty
        if not self._csv_path.exists() or self._csv_path.stat().st_size == 0:
            with self._csv_path.open("w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(self._CSV_HEADER)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def log(self, trajectory_buffer: Iterable[object]) -> None:  # noqa: D401
        """Log current statistics for *trajectory_buffer*.

        Parameters
        ----------
        trajectory_buffer:
            An *iterable* containing trajectory objects (e.g.
            ``List[RL4SysTrajectory]``).  Only the length and the Python memory
            footprint of the iterable as a whole are recorded.
        """

        buffer_list: List[object]
        if isinstance(trajectory_buffer, list):
            # Avoid copying when the input is already a list.
            buffer_list = trajectory_buffer  # type: ignore[assignment]
        else:
            # Ensure we can compute ``len`` multiple times without exhausting
            # an iterator.
            buffer_list = list(trajectory_buffer)

        buffer_len: int = len(buffer_list)
        memory_bytes: int = int(asizeof.asizeof(buffer_list))
        timestamp: str = datetime.now().isoformat(timespec="milliseconds")

        with self._lock:
            with self._csv_path.open("a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([timestamp, buffer_len, memory_bytes])

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __call__(self, trajectory_buffer: Iterable[object]) -> None:
        """Alias for :py:meth:`log` to enable *callable* style usage."""

        self.log(trajectory_buffer)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def plot_csv(
        csv_path: str | Path,
        *,
        show: bool = True,
        save_path: str | Path | None = None,
    ) -> None:
        """Plot trajectory buffer statistics stored in *csv_path*.

        Parameters
        ----------
        csv_path:
            Path to the CSV file produced by :pyclass:`TrajectoryBufferLogger`.
        show:
            Whether to display the plot in an interactive window via
            :pymeth:`matplotlib.pyplot.show`.  Defaults to ``True``.
        save_path:
            Optional path to store the generated figure.  If ``None`` no file
            is written.
        """

        import matplotlib.pyplot as plt  # Local import to avoid hard dep at runtime
        from matplotlib.dates import DateFormatter  # pylint: disable=import-error

        path: Path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        timestamps: List[datetime] = []
        buffer_lengths: List[int] = []
        memory_bytes: List[int] = []

        with path.open("r", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                try:
                    timestamps.append(datetime.fromisoformat(row["timestamp"]))
                    buffer_lengths.append(int(row["buffer_len"]))
                    memory_bytes.append(int(row["memory_bytes"]))
                except (KeyError, ValueError) as exc:
                    # Skip malformed rows but log warning to stderr
                    print(f"[TrajectoryBufferLogger] Warning: skipping row due to error: {exc}")

        if not timestamps:
            raise ValueError("CSV file does not contain any valid data rows.")

        fig, ax1 = plt.subplots(figsize=(10, 6))
        color_len = "tab:blue"
        color_mem = "tab:red"

        # Plot buffer length
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Buffer Length", color=color_len)
        ax1.plot(timestamps, buffer_lengths, color=color_len, label="Buffer Length")
        ax1.tick_params(axis="y", labelcolor=color_len)
        ax1.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()

        # Create second y-axis for memory usage
        ax2 = ax1.twinx()
        ax2.set_ylabel("Memory (MB)", color=color_mem)
        ax2.plot(
            timestamps,
            [m / 1_048_576 for m in memory_bytes],  # Convert bytes → MB
            color=color_mem,
            label="Memory (MB)",
        )
        ax2.tick_params(axis="y", labelcolor=color_mem)

        # Title and legend
        plt.title(f"Trajectory Buffer Stats — {path.stem}")
        fig.tight_layout()
        lines, labels = [], []
        for axes in (ax1, ax2):
            line, label = axes.get_legend_handles_labels()
            lines.extend(line)
            labels.extend(label)
        if lines:
            ax1.legend(lines, labels, loc="upper left")

        # Save to disk if requested
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        # -----------------------------------------------------------------
        # Print basic statistics in tabular form to STDOUT
        # -----------------------------------------------------------------
        import statistics as _stats

        def _stats_summary(values: List[int | float]):
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": _stats.mean(values),
                "median": _stats.median(values),
                "stdev": _stats.stdev(values) if len(values) > 1 else 0.0,
            }

        len_stats = _stats_summary(buffer_lengths)
        mem_stats = _stats_summary([m / 1_048_576 for m in memory_bytes])  # bytes → MB

        # Build and print table
        line_sep = "-" * 72
        print("\n" + line_sep)
        print(f"Statistics for CSV: {path}")
        print(line_sep)
        headers = ["metric", "count", "min", "max", "mean", "median", "stdev"]
        rows = [
            [
                "buffer_len",
                len_stats["count"],
                len_stats["min"],
                len_stats["max"],
                f"{len_stats['mean']:.2f}",
                len_stats["median"],
                f"{len_stats['stdev']:.2f}",
            ],
            [
                "memory_MB",
                mem_stats["count"],
                f"{mem_stats['min']:.2f}",
                f"{mem_stats['max']:.2f}",
                f"{mem_stats['mean']:.2f}",
                f"{mem_stats['median']:.2f}",
                f"{mem_stats['stdev']:.2f}",
            ],
        ]
        # Determine column widths
        col_widths = [max(len(str(row[i])) for row in ([headers] + rows)) + 2 for i in range(len(headers))]

        def _fmt_row(row):
            return "".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers)))

        print(_fmt_row(headers))
        print(line_sep)
        for row in rows:
            print(_fmt_row(row))
        print(line_sep + "\n")


# ------------------------------------------------------------------
# CLI Entrypoint
# ------------------------------------------------------------------

def main() -> None:  # noqa: D401
    """Command-line interface for plotting trajectory buffer CSV files.

    Usage
    -----
    >>> python -m rl4sys.utils.traj_buffer_log /path/to/file.csv --save plot.png

    Required positional argument ``csv_path`` specifies the CSV file to plot.
    Optional ``--no-show`` disables the interactive window and ``--save`` writes
    the figure to the given file path.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description="Plot trajectory buffer statistics from a CSV generated by TrajectoryBufferLogger.",
    )
    parser.add_argument("--csv_path", default="memtrajtest/20250716_102605/traj_buffer_log.csv", type=str, help="Path to the CSV file to plot.")
    parser.add_argument("--save", default="memtrajtest/20250716_102605/traj_buffer_log.png", type=str, help="Optional path where the resulting figure will be saved.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive window (useful when running on headless servers).")
    

    args = parser.parse_args()

    TrajectoryBufferLogger.plot_csv(
        csv_path=args.csv_path,
        show=not args.no_show,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
