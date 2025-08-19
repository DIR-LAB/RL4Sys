from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt  # type: ignore

class StepPerSecLogger:
    """Logger for recording steps/sec values to a timestamped CSV file in a dedicated folder."""

    def __init__(self, proj_name: str, output_dir: str = "speedlog") -> None:
        """
        Initialize the logger.

        Args:
            proj_name: Project name, used as the CSV file name.
            output_dir: Top-level directory for logs (default: 'speedlog').
        """
        self.proj_name: str = proj_name
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir: Path = self.output_dir / session_stamp
        self.session_dir.mkdir(exist_ok=True)
        self.csv_path: Path = self.session_dir / f"{self.proj_name}.csv"
        self._csv_file = open(self.csv_path, mode="w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=["timestamp", "steps_per_sec"])
        self._csv_writer.writeheader()

    def log(self, steps_per_sec: float) -> None:
        """
        Log a steps/sec value with the current timestamp.

        Args:
            steps_per_sec: The steps/sec value to log.
        """
        row = {
            "timestamp": datetime.now().isoformat(),
            "steps_per_sec": steps_per_sec,
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def get_csv_path(self) -> str:
        """
        Get the path to the CSV file.

        Returns:
            The CSV file path as a string.
        """
        return str(self.csv_path)

    def close(self) -> None:
        """
        Close the CSV file handle.
        """
        self._csv_file.close()

    def __enter__(self) -> StepPerSecLogger:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def plot(csv_files: List[str], output_path: str) -> None:
        """
        Plot steps/sec over time from one or more CSV files and save to output_path.

        Args:
            csv_files: List of CSV file paths to plot.
            output_path: Path to save the output plot image.
        """
        for csv_file in csv_files:
            timestamps = []
            steps_per_sec = []
            import csv
            from datetime import datetime
            with open(csv_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row.get("steps_per_sec"):
                        continue
                    try:
                        ts = datetime.fromisoformat(row["timestamp"])
                    except Exception:
                        ts = None
                    timestamps.append(ts)
                    steps_per_sec.append(float(row["steps_per_sec"]))
            label = Path(csv_file).stem
            plt.plot(range(len(steps_per_sec)), steps_per_sec, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Steps/sec")
        plt.title("Steps/sec over Epochs")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot steps/sec logs from one or more CSV files.")
    parser.add_argument("--csv_files", nargs='+', default=['speedlog/20250721_110052/job_main_rand.csv',
                                                           'speedlog/20250721_110112/job_main_infer.csv',
                                                           'speedlog/20250721_110136/job_main.csv',
                                                           'speedlog/20250721_110807/rllib_client_job_remote.csv',
                                                           'speedlog/20250721_111152/rllib_client_job_local.csv',
                                                           ], help="List of CSV files to plot")
    parser.add_argument("--output", "-o", default='speedlog/20250721_110052/job_main_rand_speedlog_20250721_110052_plot.png', help="Output image file path")
    args = parser.parse_args()
    StepPerSecLogger.plot(args.csv_files, args.output)
