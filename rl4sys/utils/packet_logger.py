"""rl4sys.utils.packet_logger

Provides a `PacketLogger` utility that can be instantiated by the server (or
any other component) to periodically persist the number of packets received
during stress-tests.

The logger
~~~~~~~~~~
* Creates a dedicated run-folder under ``packet_log/`` which is named with the
  current wall-clock timestamp (``YYYYmmdd_HHMMSS``).
* Inside that run-folder it places exactly one **CSV** file – ``<project_name>.csv`` –
  where *all* subsequent log entries are appended.
* It spawns an internal daemon thread that wakes up at a user-defined interval
  (default **5 seconds**) and writes a single line consisting of
  ``<ISO-timestamp>,<interval_count>,<total_count>,<cpu_percent>``:

      2025-07-23T18:05:00.123456,42,4242,37.5

  Where:
    * **interval_count** – number of packets received during the previous
      interval.
    * **total_count** – cumulative packets received since the logger was
      created.

Usage example
-------------
>>> logger = PacketLogger(project_name="rl4sys_server", log_interval=2.0)
>>> # somewhere in the request-handling code:
>>> logger.increment()  # call for every packet
...
>>> logger.stop()  # on shutdown

The implementation purposefully keeps *no* per-packet information in memory –
only two integers are held for counting.
"""
from __future__ import annotations

import threading
import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import Final, TextIO, List

import csv
import matplotlib.pyplot as plt
import argparse

__all__ = ["PacketLogger"]

class PacketLogger:
    """Periodically log the number of packets received to disk.

    The logger is *thread-safe* and designed to be extremely lightweight:
    calling :py:meth:`increment` merely increments two counters protected by a
    ``threading.Lock`` – no I/O happens in the hot path.  All file writes are
    delegated to a background *daemon* thread that runs every
    ``log_interval`` seconds.

    Parameters
    ----------
    project_name:
        The name of the running project – this becomes the name of the log file
        within the timestamped run folder.
    log_interval:
        How often (in seconds) to flush the counters to disk.  Defaults to
        *5 seconds*.
    base_dir:
        Base directory where the ``packet_log`` folder will be created.  This
        parameter mainly exists to aid unit-testing; leave *None* to default to
        the current working directory.
    debug:
        When *True* a couple of debug messages are printed to *stdout* – useful
        only during development.
    """

    _LOG_DIR_NAME: Final[str] = "packet_log"

    def __init__(self, *, project_name: str, log_interval: float = 5.0, base_dir: str | Path | None = None, debug: bool = False) -> None:
        if log_interval <= 0:
            raise ValueError("log_interval must be > 0")

        self._project_name: Final[str] = project_name
        self._log_interval: Final[float] = float(log_interval)
        self._debug: Final[bool] = debug

        # ------------------------------------------------------------------
        # Prepare filesystem layout – packet_log/<timestamp>/<project_name>.log
        # ------------------------------------------------------------------
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = Path(base_dir) if base_dir is not None else Path.cwd()
        self._run_dir: Path = base_path / self._LOG_DIR_NAME / timestamp
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # Use ".csv" extension to make it explicit that the file is comma-separated.
        self._log_file_path: Path = self._run_dir / f"{self._project_name}.csv"
        # Open file in *append* mode with *line buffering* so every ``write`` is
        # flushed immediately (buffering=1 works for text mode).
        self._file: TextIO = self._log_file_path.open("a", buffering=1, encoding="utf-8")

        # Write CSV header *once* when the file is empty.
        if self._file.tell() == 0:
            self._file.write("timestamp,interval_count,total_count,cpu_percent\n")
            self._file.flush()

        # Prime psutil's internal counters to get accurate non-blocking values
        psutil.cpu_percent(interval=None)

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------
        self._total_count: int = 0
        self._interval_count: int = 0
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="PacketLogger", daemon=True)
        self._thread.start()

        if self._debug:
            print(f"[PacketLogger] Started – logging to {self._log_file_path}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def increment(self, n: int = 1) -> None:
        """Register *n* additional packets.

        This method is *thread-safe* and incurs near-zero latency.

        Parameters
        ----------
        n:
            Number of packets to add to the counters.  Must be positive.
        """
        if n <= 0:
            raise ValueError("increment `n` must be positive")
        with self._lock:
            self._total_count += n
            self._interval_count += n

    def stop(self, *, wait: bool = True) -> None:
        """Gracefully shut down the background logging thread.

        It is safe to call this method multiple times.

        Parameters
        ----------
        wait:
            When *True* (default) this call blocks until the worker thread has
            exited.  When *False* it merely signals the thread to stop and
            returns immediately.
        """
        if self._stop_event.is_set():
            return  # already stopped
        self._stop_event.set()
        if wait:
            self._thread.join()
        # Ensure a final flush takes place.
        self._flush(final_flush=True)
        self._file.close()
        if self._debug:
            print("[PacketLogger] Stopped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _worker(self) -> None:  # pragma: no cover – trivial
        """Background thread body – wakes up and persists counters."""
        while not self._stop_event.wait(self._log_interval):
            self._flush()

    def _flush(self, *, final_flush: bool = False) -> None:
        """Write current counters to disk and reset the interval counter."""
        with self._lock:
            interval_count: int = self._interval_count
            self._interval_count = 0
            total_count: int = self._total_count
        if interval_count == 0 and not final_flush:
            # Avoid writing empty lines when nothing happened in the interval –
            # except for the final flush where we want to record the end state.
            return
        cpu_pct = psutil.cpu_percent(interval=None)
        timestamp_iso = datetime.now().isoformat()
        log_line = f"{timestamp_iso},{interval_count},{total_count},{cpu_pct}\n"
        self._file.write(log_line)
        # ``buffering=1`` already flushes each line, but call flush() explicitly
        # to be 100 % certain.
        self._file.flush()
        if self._debug:
            print(f"[PacketLogger] {log_line.strip()}")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------
    def __enter__(self) -> "PacketLogger":  # noqa: D401 – keep simple wording
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:  # noqa: D401
        self.stop(wait=True)
        # Do *not* suppress exceptions.
        return False

    # ------------------------------------------------------------------
    # Fallback – ensure resources are freed even if `stop` was not called.
    # ------------------------------------------------------------------
    def __del__(self) -> None:  # pragma: no cover – depends on GC timing
        try:
            self.stop(wait=False)
        except Exception:
            # Swallow all exceptions – __del__ must never raise.
            pass

    # ------------------------------------------------------------------
    # Static utility – plotting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def plot_logs(folder: str | Path, *, save_dir: str | Path | None = None, filename: str | None = None) -> None:
        """Plot packet and CPU statistics from all CSV files in *folder*.

        The function scans *folder* (non-recursively) for ``*.csv`` files
        generated by :class:`PacketLogger`.  It then creates a figure with two
        vertically-stacked sub-plots sharing the *x*-axis (wall-clock
        timestamp):

        1. *Interval packet count* – number of packets processed during each
           sampling interval.
        2. *CPU utilisation* – corresponding ``psutil.cpu_percent`` values.

        Parameters
        ----------
        folder:
            Directory containing the CSV log files.  Only files ending with
            ``.csv`` in the *top-level* of the directory are considered.
        """
        path = Path(folder)
        if not path.is_dir():
            raise FileNotFoundError(f"{path} is not a directory")

        csv_paths = sorted(p for p in path.iterdir() if p.suffix == ".csv")
        if not csv_paths:
            raise FileNotFoundError(f"No CSV files found in {path}")

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

        for csv_path in csv_paths:
            x_index: List[int] = []
            interval_counts: List[int] = []
            cpu_percents: List[float] = []

            with csv_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for idx, row in enumerate(reader):
                    try:
                        ic = int(row["interval_count"])
                        cp = float(row.get("cpu_percent", "nan"))
                    except (KeyError, ValueError):
                        continue
                    x_index.append(idx)
                    interval_counts.append(ic)
                    cpu_percents.append(cp)

            if not x_index:
                # Skip empty file.
                continue

            label = csv_path.stem  # file name without extension
            ax1.plot(x_index, interval_counts, label=label)
            ax2.plot(x_index, cpu_percents, label=label)

        ax1.set_title("Interval Packet Count (sample index)")
        ax1.set_ylabel("packets")
        ax2.set_title("CPU Utilisation (sample index)")
        ax2.set_ylabel("CPU %")

        ax1.legend(loc="upper right", fontsize="small")
        ax2.legend(loc="upper right", fontsize="small")

        # X-axis is simple integer index, no date formatting required.
        plt.tight_layout()

        # ------------------------------------------------------------------
        # Save figure if requested
        # ------------------------------------------------------------------
        if save_dir is None:
            save_dir = folder
        save_path_dir = Path(save_dir)
        save_path_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = "packet_logger_plot.png"

        fig_path = save_path_dir / filename
        fig.savefig(fig_path, dpi=150)

        # Still display interactive view for immediate feedback
        plt.show()
        print(f"[PacketLogger] Plot saved to {fig_path}")

# ---------------------------------------------------------------------------
# CLI utility
# ---------------------------------------------------------------------------


def _cli() -> None:  # pragma: no cover
    """Simple CLI for plotting PacketLogger CSV files.

    Usage:
        python -m rl4sys.utils.packet_logger <folder>
    """

    parser = argparse.ArgumentParser(description="Plot PacketLogger CSV files")
    parser.add_argument("--folder", nargs="?", default="packet_log/exp2_plt", help="Directory containing CSV logs")
    parser.add_argument("--out", type=str, default="packet_log/exp2_plt", help="Directory to save the generated plot (defaults to folder)")
    parser.add_argument("--name", type=str, default="plt_exp.png", help="Filename for the saved image (default: packet_logger_plot.png)")
    args = parser.parse_args()

    PacketLogger.plot_logs(args.folder, save_dir=args.out, filename=args.name)


# Allow `python -m rl4sys.utils.packet_logger <folder>`
if __name__ == "__main__":  # pragma: no cover
    _cli()
