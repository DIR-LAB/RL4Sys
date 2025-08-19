"""Memory profiler utility using psutil for comprehensive memory monitoring.

This module provides a MemoryProfiler class that can monitor and log various
memory metrics including USS, PSS, RSS, and actual memory usage for processes.
Supports both one-time profiling and continuous background monitoring.
"""

import os
import psutil
import json
import csv
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
import matplotlib.pyplot as plt  # type: ignore


class MemoryProfiler:
    """A comprehensive memory profiler using psutil to monitor memory usage.
    
    This class provides functionality to collect various memory metrics including
    USS (Unique Set Size), PSS (Proportional Set Size), RSS (Resident Set Size),
    and actual memory usage. Results can be saved to timestamped files in a
    specified directory. Supports continuous background monitoring via threading.
    """

    def __init__(self, proj_name: str, output_dir: str = "memtest", 
                 log_interval: float = 3.0) -> None:
        """Initialize the MemoryProfiler.
        
        Args:
            proj_name: The project name to include in output filenames.
            output_dir: The directory to save memory profiling results.
                       Defaults to "memtest".
            log_interval: Interval in seconds for background logging.
                         Defaults to 3.0 seconds.
        """
        self.proj_name: str = proj_name
        self.output_dir: Path = Path(output_dir)
        self.log_interval: float = log_interval
        # ------------------------------------------------------------------
        # Streaming CSV setup (avoid holding data in memory)
        # ------------------------------------------------------------------
        self._csv_file_handle: Optional[Any] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_filepath: Optional[Path] = None
        self._header_written: bool = False

        # Incremental statistics for summary (RSS only to keep it simple)
        self._cnt: int = 0
        self._rss_sum: int = 0
        self._rss_min: Optional[int] = None
        self._rss_max: Optional[int] = None
        
        # Threading attributes
        self._profiling_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._lock: threading.Lock = threading.Lock()
        self._is_running: bool = False
        self._target_pid: Optional[int] = None
        self._include_system: bool = True
        self._callback: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        # Create a run-specific timestamped folder to avoid mixing files
        ts_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir: Path = self.output_dir / ts_folder
        self._session_dir.mkdir(exist_ok=True)

    def get_process_memory_info(self, pid: Optional[int] = None) -> Dict[str, Union[int, float]]:
        """Get comprehensive memory information for a specific process.
        
        Args:
            pid: Process ID to monitor. If None, monitors current process.
            
        Returns:
            Dictionary containing memory metrics in bytes.
            
        Raises:
            psutil.NoSuchProcess: If the specified process doesn't exist.
            psutil.AccessDenied: If access to process information is denied.
        """
        try:
            process = psutil.Process(pid) if pid else psutil.Process()
            
            # Get memory info
            memory_info = process.memory_info()
            
            # Get memory full info (includes USS, PSS if available)
            try:
                memory_full_info = process.memory_full_info()
                uss = memory_full_info.uss
                pss = memory_full_info.pss
            except (AttributeError, psutil.AccessDenied):
                # Fallback if USS/PSS not available on this platform
                uss = 0
                pss = 0
            
            # Get memory percentage
            memory_percent = process.memory_percent()
            
            # Get system memory info for context
            system_memory = psutil.virtual_memory()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'pid': process.pid,
                'process_name': process.name(),
                'rss': memory_info.rss,  # Resident Set Size
                'vms': memory_info.vms,  # Virtual Memory Size
                'uss': uss,              # Unique Set Size
                'pss': pss,              # Proportional Set Size
                'memory_percent': memory_percent,
                'actual_memory_bytes': memory_info.rss,  # RSS is the actual physical memory
                'system_total_memory': system_memory.total,
                'system_available_memory': system_memory.available,
                'system_memory_percent': system_memory.percent
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            raise e

    def get_system_memory_info(self) -> Dict[str, Union[int, float]]:
        """Get comprehensive system-wide memory information.
        
        Returns:
            Dictionary containing system memory metrics.
        """
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_memory': virtual_memory.total,
            'available_memory': virtual_memory.available,
            'used_memory': virtual_memory.used,
            'free_memory': virtual_memory.free,
            'memory_percent': virtual_memory.percent,
            'buffers': getattr(virtual_memory, 'buffers', 0),
            'cached': getattr(virtual_memory, 'cached', 0),
            'shared': getattr(virtual_memory, 'shared', 0),
            'swap_total': swap_memory.total,
            'swap_used': swap_memory.used,
            'swap_free': swap_memory.free,
            'swap_percent': swap_memory.percent
        }

    def profile_memory(self, pid: Optional[int] = None, include_system: bool = True) -> Dict[str, Any]:
        """Profile memory usage for a process and optionally system-wide.
        
        Args:
            pid: Process ID to monitor. If None, monitors current process.
            include_system: Whether to include system-wide memory information.
            
        Returns:
            Dictionary containing all collected memory metrics.
        """
        profile_data = {}
        
        # Get process memory info
        try:
            profile_data['process'] = self.get_process_memory_info(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            profile_data['process_error'] = str(e)
        
        # Get system memory info if requested
        if include_system:
            profile_data['system'] = self.get_system_memory_info()
 
        # ------------------------------------------------------------------
        # Stream the data directly to CSV to avoid memory accumulation
        # ------------------------------------------------------------------
        flattened = self._flatten_profile_data(profile_data)

        with self._lock:
            # Lazily create CSV writer and file on first write
            if self._csv_writer is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.proj_name}_memory_profile_{timestamp}.csv"
                self._csv_filepath = self._session_dir / filename
                self._csv_file_handle = open(self._csv_filepath, "w", newline="")
                self._csv_writer = csv.DictWriter(self._csv_file_handle, fieldnames=list(flattened.keys()))
                self._csv_writer.writeheader()

            self._csv_writer.writerow(flattened)
            self._csv_file_handle.flush()

            # Update statistics
            rss = flattened.get('process_rss')
            if isinstance(rss, (int, float)):
                self._cnt += 1
                self._rss_sum += rss
                if self._rss_min is None or rss < self._rss_min:
                    self._rss_min = rss
                if self._rss_max is None or rss > self._rss_max:
                    self._rss_max = rss
 
        return profile_data

    def _background_profiling_loop(self) -> None:
        """Internal method for background profiling loop."""
        while not self._stop_event.is_set():
            try:
                # Profile memory
                data = self.profile_memory(self._target_pid, self._include_system)
                
                # Call callback if provided
                if self._callback:
                    self._callback(data)
                    
            except Exception as e:
                # Log error but continue profiling
                error_data = {
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                with self._lock:
                    if self._csv_writer is None:
                        # Initialize writer with basic error fields
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{self.proj_name}_memory_profile_{timestamp}.csv"
                        self._csv_filepath = self._session_dir / filename
                        self._csv_file_handle = open(self._csv_filepath, "w", newline="")
                        self._csv_writer = csv.DictWriter(self._csv_file_handle, fieldnames=list(error_data.keys()))
                        self._csv_writer.writeheader()
                    self._csv_writer.writerow(error_data)
                    self._csv_file_handle.flush()
            
            # Wait for the specified interval or until stop event
            self._stop_event.wait(timeout=self.log_interval)

    def start_background_profiling(self, pid: Optional[int] = None, 
                                 include_system: bool = True,
                                 callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """Start background memory profiling in a separate thread.
        
        Args:
            pid: Process ID to monitor. If None, monitors current process.
            include_system: Whether to include system-wide memory information.
            callback: Optional callback function called with each profile result.
            
        Raises:
            RuntimeError: If background profiling is already running.
        """
        if self._is_running:
            raise RuntimeError("Background profiling is already running")
        
        self._target_pid = pid
        self._include_system = include_system
        self._callback = callback
        self._stop_event.clear()
        self._is_running = True
        
        # Start the profiling thread
        self._profiling_thread = threading.Thread(
            target=self._background_profiling_loop,
            name=f"MemoryProfiler-{self.proj_name}",
            daemon=True
        )
        self._profiling_thread.start()

    def stop_background_profiling(self, timeout: Optional[float] = 5.0) -> bool:
        """Stop background memory profiling.
        
        Args:
            timeout: Maximum time to wait for thread to stop. None means wait forever.
            
        Returns:
            True if profiling was stopped successfully, False if timeout occurred.
        """
        if not self._is_running:
            return True
        
        # Signal the thread to stop
        self._stop_event.set()
        
        # Wait for thread to finish
        if self._profiling_thread and self._profiling_thread.is_alive():
            self._profiling_thread.join(timeout=timeout)
            
            if self._profiling_thread.is_alive():
                # Thread didn't stop within timeout
                return False
        
        self._is_running = False
        self._profiling_thread = None

        # Ensure data flushed but keep file open for potential further use
        with self._lock:
            if self._csv_file_handle:
                self._csv_file_handle.flush()

        return True

    def is_profiling(self) -> bool:
        """Check if background profiling is currently running.
        
        Returns:
            True if background profiling is active, False otherwise.
        """
        return self._is_running and (
            self._profiling_thread is not None and self._profiling_thread.is_alive()
        )

    def set_log_interval(self, interval: float) -> None:
        """Set the logging interval for background profiling.
        
        Args:
            interval: New interval in seconds.
            
        Note:
            Changes take effect for the next profiling cycle.
        """
        if interval <= 0:
            raise ValueError("Log interval must be positive")
        self.log_interval = interval

    def save_to_json(self, data: Optional[List[Dict[str, Any]]] = None, 
                     filename_suffix: str = "") -> str:
        """JSON saving is disabled in streaming mode to avoid extra memory overhead."""
        raise NotImplementedError("Streaming MemoryProfiler supports CSV output only.")

    def save_to_csv(self, data: Optional[List[Dict[str, Any]]] = None,
                    filename_suffix: str = "") -> str:
        """Return the path to the CSV file (streaming mode)."""
        with self._lock:
            if self._csv_file_handle is None:
                raise ValueError("No data has been collected yet; CSV file not created.")

            # Flush to ensure all data is written
            self._csv_file_handle.flush()

            # If a suffix is requested, copy the file under a new name
            if filename_suffix:
                new_filepath = self._csv_filepath.with_name(
                    self._csv_filepath.stem + f"_{filename_suffix}" + self._csv_filepath.suffix
                )
                import shutil
                shutil.copy(self._csv_filepath, new_filepath)
                return str(new_filepath)

            return str(self._csv_filepath)

    def clear_data(self) -> None:
        """Reset internal statistics and close the CSV file (if open)."""
        with self._lock:
            if self._csv_file_handle:
                self._csv_file_handle.close()
                self._csv_file_handle = None
                self._csv_writer = None
                self._csv_filepath = None
                self._header_written = False
            self._cnt = 0
            self._rss_sum = 0
            self._rss_min = None
            self._rss_max = None

    def get_data_count(self) -> int:
        """Get the number of collected memory data points."""
        with self._lock:
            return self._cnt

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of collected memory data using incremental stats."""
        with self._lock:
            if self._cnt == 0:
                return {'message': 'No data collected'}

            avg_rss = self._rss_sum / self._cnt if self._cnt else 0

            return {
                'total_measurements': self._cnt,
                'project_name': self.proj_name,
                'log_interval': self.log_interval,
                'rss_stats': {
                    'min_bytes': self._rss_min,
                    'max_bytes': self._rss_max,
                    'avg_bytes': avg_rss,
                    'min_mb': self._rss_min / (1024 * 1024) if self._rss_min is not None else None,
                    'max_mb': self._rss_max / (1024 * 1024) if self._rss_max is not None else None,
                    'avg_mb': avg_rss / (1024 * 1024),
                }
            }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures background profiling is stopped."""
        if self.is_profiling():
            self.stop_background_profiling()

    def _flatten_profile_data(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested profiling dict into a single-level dict for CSV."""
        flat_entry: Dict[str, Any] = {}

        # Process data
        if 'process' in entry:
            for key, value in entry['process'].items():
                flat_entry[f'process_{key}'] = value

        # System data
        if 'system' in entry:
            for key, value in entry['system'].items():
                flat_entry[f'system_{key}'] = value

        # Errors
        if 'process_error' in entry:
            flat_entry['process_error'] = entry['process_error']
        if 'profiling_error' in entry:
            for key, value in entry['profiling_error'].items():
                flat_entry[f'error_{key}'] = value

        return flat_entry


def quick_memory_snapshot(proj_name: str, pid: Optional[int] = None, 
                         save_format: str = "csv") -> str:
    """Take a quick memory snapshot and save it immediately.
    
    Args:
        proj_name: Project name for the output file.
        pid: Process ID to monitor. If None, monitors current process.
        save_format: Format to save the data ("json" or "csv").
        
    Returns:
        Path to the saved file.
    """
    profiler = MemoryProfiler(proj_name)
    profiler.profile_memory(pid)
    
    # Streaming mode supports CSV only
    return profiler.save_to_csv(filename_suffix="snapshot")


def profile_for_duration(proj_name: str, duration: float, 
                        log_interval: float = 3.0, pid: Optional[int] = None,
                        save_format: str = "csv") -> str:
    """Profile memory for a specific duration and save results.
    
    Args:
        proj_name: Project name for the output file.
        duration: Duration to profile in seconds.
        log_interval: Interval between measurements in seconds.
        pid: Process ID to monitor. If None, monitors current process.
        save_format: Format to save the data ("json" or "csv").
        
    Returns:
        Path to the saved file.
    """
    with MemoryProfiler(proj_name, log_interval=log_interval) as profiler:
        profiler.start_background_profiling(pid=pid)
        time.sleep(duration)
        profiler.stop_background_profiling()
        
        return profiler.save_to_csv(filename_suffix=f"duration_{duration}s")


def plot_memory_profiles(csv_files: List[str], output_path: str) -> None:
    """
    Plot process_rss over time from one or more CSV files and save to output_path.

    Args:
        csv_files: List of CSV file paths to plot.
        output_path: Path to save the output plot image.
    """
    for csv_file in csv_files:
        rss_values = []
        import csv
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("process_rss"):
                    continue
                try:
                    rss = float(row["process_rss"])
                except Exception:
                    continue
                rss_values.append(rss / (1024 * 1024))  # Convert to MB
        label = Path(csv_file).stem
        plt.plot(range(len(rss_values)), rss_values, label=label)
    plt.xlabel("Sample")
    plt.ylabel("Process RSS (MB)")
    plt.title("Process RSS over Time")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot process RSS from one or more memory profiler CSV files.")
    parser.add_argument("--csv_files", nargs='+', default=['memtest/20250721_110052/job_main_rand 100 traj send_memory_profile_20250721_110052.csv',
                                                           'memtest/20250721_110112/job_main_infer 100 traj send_memory_profile_20250721_110112.csv',
                                                           'memtest/20250721_110136/job_main 1 traj send_memory_profile_20250721_110136.csv',
                                                           'memtest/20250721_110807/rllib_client_job_remote_memory_profile_20250721_110807.csv',
                                                           'memtest/20250721_111152/rllib_client_job_local_memory_profile_20250721_111152.csv',
                                                           ], help="List of CSV files to plot")
    parser.add_argument("--output", "-o", default='memtest/20250721_111152/rllib_client_job_local_memory_profile_20250721_111152.png', help="Output image file path")
    args = parser.parse_args()
    plot_memory_profiles(args.csv_files, args.output)
