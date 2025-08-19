from __future__ import annotations

"""Streaming CPU profiler utility using psutil.

This module provides a :class:`CPUProfiler` for real-time monitoring of CPU
usage of a single process (and optionally the whole system).  The profiler is
optimised for long-running experiments where holding all measurements in memory
would be undesirable: every sample is **immediately** appended to a CSV file on
disk.

Key features
------------
1. Streaming CSV logging – no in-memory accumulation.
2. Automatic directory management – results are written to ``./cputest/<timestamp>/``.
3. Thread-based background profiling with a configurable sampling interval
   (default **3 seconds**).
4. Simple, functional API wrappers (`quick_cpu_snapshot`,
   `profile_for_duration`) for convenience.

Example
~~~~~~~
```python
from rl4sys.utils.cpu_prof import profile_for_duration

# Collect CPU stats of *this* process for 60 s and sample every 2 s.
csv_path = profile_for_duration("my_experiment", duration=60, log_interval=2)
print(f"Results stored at {csv_path}")
```
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import csv
import threading
import time

import psutil

__all__ = [
    "CPUProfiler",
    "quick_cpu_snapshot",
    "profile_for_duration",
    "plot_cpu_profile",
]


class CPUProfiler:  # pylint: disable=too-many-instance-attributes
    """Streaming CPU profiler based on :pymod:`psutil`.

    Parameters
    ----------
    proj_name
        A short label used in the generated filename so that multiple
        experiments running in parallel do not clash.
    output_dir
        Top-level directory where *all* CPU profiling sessions are stored.  The
        default is ``"cputest"`` (created if missing).
    log_interval
        Sampling interval in *seconds* for background profiling.  A value of
        ``3.0`` seconds is used if none is supplied.
    """

    def __init__(self, proj_name: str, *, output_dir: str = "cputest", log_interval: float = 3.0) -> None:  # noqa: D401,E501
        self.proj_name: str = proj_name
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Each *run* has its own timestamped sub-directory – keeps things tidy
        # when multiple experiments are executed back-to-back.
        session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir: Path = self.output_dir / session_stamp
        self._session_dir.mkdir(exist_ok=True)

        self.log_interval: float = log_interval

        # Streaming CSV handles – lazily opened on first write to avoid empty
        # files if the user never collects any data.
        self._csv_handle: Optional[Any] = None  # pylint: disable=invalid-name
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_path: Optional[Path] = None

        # Stats for a lightweight summary (CPU percentages)
        self._cnt: int = 0
        self._cpu_sum: float = 0.0
        self._cpu_min: Optional[float] = None
        self._cpu_max: Optional[float] = None

        # Background thread machinery
        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._lock: threading.Lock = threading.Lock()
        self._is_running: bool = False
        # Single cached process instance (like system_monitor.py)
        self._process: Optional[psutil.Process] = None
        self._target_pid: Optional[int] = None
        self._include_system: bool = True
        self._callback: Optional[Callable[[Dict[str, Any]], None]] = None

    # ---------------------------------------------------------------------
    # Core data collection helpers
    # ---------------------------------------------------------------------
    def get_process_cpu_info(self, pid: Optional[int] = None) -> Dict[str, Any]:
        """Return current CPU statistics of a *single* process.

        The function captures:
        * `cpu_percent` – instantaneous CPU utilisation of the process.
        * `cpu_times_user` / `cpu_times_system` – cumulative CPU times.
        * `num_threads` – number of threads belonging to the process.

        Parameters
        ----------
        pid
            PID of the target process.  If *None* (default) the *current*
            Python process is inspected.
        """
        # Get or create cached process instance
        if self._process is None or (pid is not None and pid != self._target_pid):
            self._target_pid = pid
            self._process = psutil.Process(pid) if pid is not None else psutil.Process()
            # Prime the CPU counter (first call returns 0.0)
            self._process.cpu_percent()

        # Get CPU metrics (subsequent calls give meaningful values)
        # According to psutil docs: values > 100% are normal for multi-threaded processes
        cpu_percent_raw = self._process.cpu_percent()
        cpu_times = self._process.cpu_times()

        # Get logical CPU count for normalization
        logical_cpus = psutil.cpu_count(logical=True) or 1
        cpu_percent_per_core = cpu_percent_raw / logical_cpus

        return {
            "timestamp": datetime.now().isoformat(),
            "pid": self._process.pid,
            "process_name": self._process.name(),
            # Raw CPU percentage (can exceed 100% for multi-threaded processes)
            "cpu_percent": cpu_percent_raw,
            # CPU percentage normalized per logical core (0-100% range)
            "cpu_percent_per_core": cpu_percent_per_core,
            # Number of logical CPUs for reference
            "logical_cpu_count": logical_cpus,
            "cpu_times_user": cpu_times.user,
            "cpu_times_system": cpu_times.system,
            "num_threads": self._process.num_threads(),
        }

    def get_system_cpu_info(self) -> Dict[str, Any]:
        """Return system-wide CPU utilisation statistics."""
        # Get system-wide CPU percentage
        system_cpu_percent = psutil.cpu_percent(interval=None)
        
        # Get per-CPU utilization (blocking call for accuracy)
        try:
            # Use a short interval to get per-CPU data without blocking too long
            per_cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        except Exception:
            # Fallback if per-CPU fails
            per_cpu_percent = []

        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": system_cpu_percent,
            "cpu_percent_per_core": per_cpu_percent,
            "cpu_count": psutil.cpu_count(logical=True),
            # Load average is not available on Windows – fall back accordingly.
            "load_avg_1m": (lambda: psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0.0)(),  # noqa: E501
        }

    def profile_cpu(self, pid: Optional[int] = None, include_system: bool = True) -> Dict[str, Any]:  # noqa: D401,E501
        """Collect a CPU usage sample and **stream it directly** to CSV.

        Notes
        -----
        Unlike typical profilers this method **does not** accumulate data in
        memory – each sample is flushed to disk immediately, enabling profiling
        of very long sessions without OOM concerns.
        """
        entry: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Process-level stats (with basic exception handling)
        # ------------------------------------------------------------------
        try:
            entry["process"] = self.get_process_cpu_info(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:  # pragma: no cover
            entry["process_error"] = str(exc)

        # ------------------------------------------------------------------
        # System-level stats (optional)
        # ------------------------------------------------------------------
        if include_system:
            entry["system"] = self.get_system_cpu_info()

        # ------------------------------------------------------------------
        # Flatten + stream to CSV
        # ------------------------------------------------------------------
        flat = self._flatten_entry(entry)
        self._write_row(flat)

        # Update incremental summary statistics (both raw and per-core process CPU)
        proc_cpu_raw = flat.get("process_cpu_percent")
        proc_cpu_per_core = flat.get("process_cpu_percent_per_core")
        
        if isinstance(proc_cpu_raw, (int, float)):
            self._cnt += 1
            self._cpu_sum += proc_cpu_raw
            self._cpu_min = proc_cpu_raw if self._cpu_min is None else min(self._cpu_min, proc_cpu_raw)
            self._cpu_max = proc_cpu_raw if self._cpu_max is None else max(self._cpu_max, proc_cpu_raw)
            
            # Also track per-core statistics
            if isinstance(proc_cpu_per_core, (int, float)):
                if not hasattr(self, '_cpu_per_core_sum'):
                    self._cpu_per_core_sum = 0.0
                    self._cpu_per_core_min = None
                    self._cpu_per_core_max = None
                self._cpu_per_core_sum += proc_cpu_per_core
                self._cpu_per_core_min = proc_cpu_per_core if self._cpu_per_core_min is None else min(self._cpu_per_core_min, proc_cpu_per_core)
                self._cpu_per_core_max = proc_cpu_per_core if self._cpu_per_core_max is None else max(self._cpu_per_core_max, proc_cpu_per_core)

        return entry

    # ------------------------------------------------------------------
    # Streaming CSV helpers
    # ------------------------------------------------------------------
    def _write_row(self, row: Dict[str, Any]) -> None:
        """Write *row* to CSV – lazily initialise file on first call."""
        with self._lock:
            if self._csv_writer is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.proj_name}_cpu_profile_{ts}.csv"
                self._csv_path = self._session_dir / filename
                self._csv_handle = open(self._csv_path, "w", newline="")
                self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=list(row.keys()))
                self._csv_writer.writeheader()
            # Write + flush so that data is durable even if the process crashes.
            self._csv_writer.writerow(row)
            self._csv_handle.flush()

    # ------------------------------------------------------------------
    # Background profiling machinery
    # ------------------------------------------------------------------
    def _profiling_loop(self) -> None:
        """Internal thread function – runs until :pyattr:`_stop_event` is set."""
        while not self._stop_event.is_set():
            try:
                data = self.profile_cpu(self._target_pid, self._include_system)
                if self._callback is not None:
                    self._callback(data)
            except Exception as exc:  # pylint: disable=broad-except
                # Log straight into the CSV so that we do not lose the error.
                err_row = {
                    "timestamp": datetime.now().isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                self._write_row(err_row)

            # Wait for next interval (with early exit support)
            self._stop_event.wait(timeout=self.log_interval)

    def start_background_profiling(
        self,
        *,
        pid: Optional[int] = None,
        include_system: bool = True,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """Start continuous background profiling in a *separate* thread."""
        if self._is_running:
            raise RuntimeError("Background CPU profiling is already running")

        self._target_pid = pid
        self._include_system = include_system
        self._callback = callback

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._profiling_loop, daemon=True)
        self._thread.start()
        self._is_running = True

    def stop_background_profiling(self, *, timeout: Optional[float] = 5.0) -> bool:  # noqa: D401,E501
        """Signal the background thread to terminate and *optionally* join it.

        Returns
        -------
        bool
            *True* if the thread terminated gracefully, *False* otherwise.
        """
        if not self._is_running:
            return True  # Nothing to do.

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

        stopped_normally = self._thread is None or not self._thread.is_alive()
        self._is_running = False
        return stopped_normally

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def set_log_interval(self, interval: float) -> None:
        """Change the sampling interval *during* a run."""
        if interval <= 0:
            raise ValueError("Log interval must be positive")
        self.log_interval = interval

    def get_csv_path(self) -> str:
        """Return the path of the CSV file (raises if no data collected)."""
        if self._csv_path is None:
            raise ValueError("No data has been collected yet; CSV file not created.")
        return str(self._csv_path)

    def get_cpu_summary(self) -> Dict[str, Any]:
        """Return min/max/average of *process* CPU percent based on streamed data."""
        if self._cnt == 0:
            return {"message": "No data collected"}

        avg_cpu = self._cpu_sum / self._cnt
        summary = {
            "total_measurements": self._cnt,
            "log_interval": self.log_interval,
            "process_cpu_stats": {
                "min_percent": self._cpu_min,
                "max_percent": self._cpu_max,
                "avg_percent": avg_cpu,
            },
        }
        
        # Add per-core statistics if available
        if hasattr(self, '_cpu_per_core_sum') and self._cnt > 0:
            avg_cpu_per_core = self._cpu_per_core_sum / self._cnt
            summary["process_cpu_per_core_stats"] = {
                "min_percent": self._cpu_per_core_min,
                "max_percent": self._cpu_per_core_max,
                "avg_percent": avg_cpu_per_core,
            }
        
        return summary

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------
    def __enter__(self) -> "CPUProfiler":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: D401
        if self._is_running:
            self.stop_background_profiling()

    # ------------------------------------------------------------------
    # Utility – flatten nested dicts for CSV
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        if "process" in entry:
            for key, val in entry["process"].items():
                flat[f"process_{key}"] = val
        if "system" in entry:
            for key, val in entry["system"].items():
                flat[f"system_{key}"] = val
        if "process_error" in entry:
            flat["process_error"] = entry["process_error"]
        if "profiling_error" in entry:
            for key, val in entry["profiling_error"].items():
                flat[f"error_{key}"] = val
        return flat


# ======================================================================
# Functional API wrappers – sometimes you just need one-liners
# ======================================================================

def quick_cpu_snapshot(
    proj_name: str,
    *,
    pid: Optional[int] = None,
) -> str:
    """Capture a *single* CPU sample and immediately write it to a CSV file.

    Returns the path of the created file so the caller can post-process it.
    """
    profiler = CPUProfiler(proj_name)
    profiler.profile_cpu(pid)
    return profiler.get_csv_path()


def profile_for_duration(
    proj_name: str,
    *,
    duration: float,
    pid: Optional[int] = None,
    log_interval: float = 3.0,
) -> str:
    """Profile CPU usage for *duration* seconds and save the streamed CSV.

    The function is a blocking helper intended for scripting / CLI use-cases.
    """
    with CPUProfiler(proj_name, log_interval=log_interval) as profiler:
        profiler.start_background_profiling(pid=pid)
        time.sleep(duration)
        profiler.stop_background_profiling()
        return profiler.get_csv_path()

# ======================================================================
# Data visualisation helper
# ======================================================================

def plot_cpu_profile(csv_path: str, output_path: Optional[str] = None, *, include_system: bool = True) -> None:
    """Plot CPU utilisation over time from a streamed CSV file.

    Parameters
    ----------
    csv_path
        Path to the CSV file produced by :class:`CPUProfiler`.
    output_path
        Path where the resulting PNG file will be stored.  If *None*, the plot
        is saved alongside *csv_path* with a ``_plot.png`` suffix.
    include_system
        When *True* (default) the system-wide CPU utilisation curve is also
        plotted (if available in the CSV).  Otherwise only the process curve is
        shown.
    """
    import csv
    from datetime import datetime

    # Local import so that the heavy dependency is only required for plotting.
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    timestamps: List[datetime] = []
    process_cpu_raw: List[float] = []
    process_cpu_per_core: List[float] = []
    system_cpu: List[float] = []
    system_cpu_per_core_data: List[List[float]] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows without process_cpu_percent (e.g., error rows)
            if not row.get("process_cpu_percent"):
                continue
            try:
                ts_str = row["process_timestamp"] or row.get("timestamp", "")
                # Fallback to index if timestamp parsing fails
                ts = datetime.fromisoformat(ts_str) if ts_str else None
            except (ValueError, KeyError):
                ts = None
            timestamps.append(ts or datetime.now())  # dummy value if parsing fails
            
            # Process CPU data
            process_cpu_raw.append(float(row["process_cpu_percent"]))
            if row.get("process_cpu_percent_per_core"):
                process_cpu_per_core.append(float(row["process_cpu_percent_per_core"]))
            else:
                process_cpu_per_core.append(float("nan"))
            
            # System CPU data
            if include_system and row.get("system_cpu_percent"):
                system_cpu.append(float(row["system_cpu_percent"]))
            elif include_system:
                system_cpu.append(float("nan"))
            
            # Per-core system CPU data
            if include_system and row.get("system_cpu_percent_per_core"):
                try:
                    # Parse the per-core data (stored as string representation of list)
                    per_core_str = row["system_cpu_percent_per_core"]
                    if per_core_str.startswith("[") and per_core_str.endswith("]"):
                        per_core_values = [float(x.strip()) for x in per_core_str[1:-1].split(",")]
                        system_cpu_per_core_data.append(per_core_values)
                    else:
                        system_cpu_per_core_data.append([])
                except (ValueError, AttributeError):
                    system_cpu_per_core_data.append([])
            elif include_system:
                system_cpu_per_core_data.append([])

    if not timestamps:
        raise ValueError("No valid data found in CSV – cannot generate plot.")

    # Use sample index for x-axis
    x_axis = list(range(len(timestamps)))

    # Create figure with two line subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Subplot 1 – raw process CPU percent (can exceed 100%)
    ax1.plot(x_axis, process_cpu_raw, label="Process CPU % (raw)", color="crimson", linewidth=1.8)
    ax1.set_ylabel("CPU % (raw)")
    ax1.set_title("Process CPU utilisation (raw)")
    ax1.grid(True, alpha=0.3)

    # Subplot 2 – per-core normalised process CPU percent (0–100)
    ax2.plot(x_axis, process_cpu_per_core, label="Process CPU % (per-core)", color="steelblue", linewidth=1.8)
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("CPU % per core")
    ax2.set_title("Process CPU utilisation (normalised per logical CPU)")
    ax2.grid(True, alpha=0.3)

    # Legends – placed outside to avoid overlap if many points
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    plt.tight_layout()

    if output_path is None:
        output_path = str(Path(csv_path).with_suffix("_plot.png"))
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compare_cpu_profiles(
    csv_path1: str,
    csv_path2: str,
    csv_path3: str,
    output_path: Optional[str] = None,
    *,
    include_system: bool = True,
) -> None:
    """Compare CPU utilisation over time from three streamed CSV files.

    Parameters
    ----------
    csv_path1, csv_path2, csv_path3
        Paths to the CSV files produced by :class:`CPUProfiler`.
    output_path
        Path where the resulting PNG file will be stored. If *None*, the plot
        is saved alongside *csv_path1* with a ``_compare_plot.png`` suffix.
    include_system
        When *True* (default) the system-wide CPU utilisation curve is also
        plotted (if available in the CSVs). Currently, only process curves are compared.
    """
    import csv
    from datetime import datetime
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
    from pathlib import Path

    def load_cpu_data(csv_path: str) -> tuple[list[datetime], list[float], list[float]]:
        timestamps: list[datetime] = []
        process_cpu_raw: list[float] = []
        process_cpu_per_core: list[float] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("process_cpu_percent"):
                    continue
                try:
                    ts_str = row["process_timestamp"] or row.get("timestamp", "")
                    ts = datetime.fromisoformat(ts_str) if ts_str else None
                except (ValueError, KeyError):
                    ts = None
                timestamps.append(ts or datetime.now())
                process_cpu_raw.append(float(row["process_cpu_percent"]))
                if row.get("process_cpu_percent_per_core"):
                    process_cpu_per_core.append(float(row["process_cpu_percent_per_core"]))
                else:
                    process_cpu_per_core.append(float("nan"))
        return timestamps, process_cpu_raw, process_cpu_per_core

    # Load data from all three CSVs
    data1 = load_cpu_data(csv_path1)
    data2 = load_cpu_data(csv_path2)
    data3 = load_cpu_data(csv_path3)

    # Use sample index for x-axis
    x1 = list(range(len(data1[0])))
    x2 = list(range(len(data2[0])))
    x3 = list(range(len(data3[0])))

    # Labels for legend (use file name)
    label1 = Path(csv_path1).stem
    label2 = Path(csv_path2).stem
    label3 = Path(csv_path3).stem

    # Create figure with two line subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Subplot 1 – raw process CPU percent (can exceed 100%)
    ax1.plot(x1, data1[1], label=f"{label1}", color="crimson", linewidth=1.8)
    ax1.plot(x2, data2[1], label=f"{label2}", color="seagreen", linewidth=1.8)
    ax1.plot(x3, data3[1], label=f"{label3}", color="darkorange", linewidth=1.8)
    ax1.set_ylabel("CPU % (raw)")
    ax1.set_title("Process CPU utilisation (raw) – Comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Subplot 2 – per-core normalised process CPU percent (0–100)
    ax2.plot(x1, data1[2], label=f"{label1}", color="crimson", linewidth=1.8)
    ax2.plot(x2, data2[2], label=f"{label2}", color="seagreen", linewidth=1.8)
    ax2.plot(x3, data3[2], label=f"{label3}", color="darkorange", linewidth=1.8)
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("CPU % per core")
    ax2.set_title("Process CPU utilisation (normalised per logical CPU) – Comparison")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()

    if output_path is None:
        output_path = str(Path(csv_path1).with_suffix("_compare_plot.png"))
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ======================================================================
# CLI wrapper
# ======================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot or compare CPU profile CSVs produced by CPUProfiler.")
    parser.add_argument("--csv_file", default="cputest/20250718_170955/job_main 10 traj send_cpu_profile_20250718_170955.csv", help="Path to CPU profile CSV file")
    parser.add_argument("--output", "-o", dest="output", default="cputest/20250721_100700/rllib_client_job_remote_cpu_profile_20250721_100700.png", help="Where to save the PNG plot")
    parser.add_argument("--no-system", action="store_false", help="Do not include system-wide CPU curve")
    parser.add_argument(
        "--compare",
        nargs=3,
        default=['cputest/20250721_100700/rllib_client_job_remote_cpu_profile_20250721_100700.csv', 
                 'cputest/20250721_100321/rllib_client_job_cpu_profile_20250721_100321.csv', 
                 'cputest/20250721_091912/job_main 10 traj send_cpu_profile_20250721_091912.csv'],
        metavar=("CSV1", "CSV2", "CSV3"),
        help="Compare three CPU profile CSV files on the same plot. Overrides --csv_file.",
    )
    args = parser.parse_args()

    if args.compare:
        compare_output = args.output or "cpu_profile_compare_plot.png"
        compare_cpu_profiles(
            args.compare[0],
            args.compare[1],
            args.compare[2],
            output_path=compare_output,
            include_system=not args.no_system,
        )
    else:
        plot_cpu_profile(args.csv_file, output_path=args.output, include_system=not args.no_system)
