"""
System health monitoring utilities for RL4Sys.

This module provides tools for monitoring memory usage, system resources,
and performance metrics across the distributed training framework.
"""

import os
import time
import psutil
import threading
import csv
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from rl4sys.utils.util import StructuredLogger

@dataclass
class SystemMetrics:
    """Container for system performance metrics."""
    memory_percent: float
    memory_available_gb: float
    memory_used_gb: float
    cpu_percent: float
    disk_usage_percent: float
    process_memory_mb: float
    process_cpu_percent: float
    network_bytes_sent: float
    network_bytes_recv: float
    network_packets_sent: float
    network_packets_recv: float
    timestamp: float

class SystemMonitor:
    """
    System health monitoring for RL4Sys components.
    
    Provides memory usage tracking, CPU monitoring, network bandwidth monitoring,
    and performance metrics with configurable logging intervals and thresholds.
    """
    
    def __init__(self, 
                 name: str = "SystemMonitor",
                 log_interval: float = 30.0,
                 memory_threshold: float = 85.0,
                 cpu_threshold: float = 90.0,
                 debug: bool = False,
                 save_to_file: bool = False,
                 project_name: str = "default_project"):
        """
        Initialize system monitor.
        
        Args:
            name: Monitor instance name for logging
            log_interval: Seconds between automatic metric logging
            memory_threshold: Memory usage percentage threshold for warnings
            cpu_threshold: CPU usage percentage threshold for warnings
            debug: Enable debug logging
            save_to_file: Whether to save metrics to file
            project_name: Project name for organizing saved files
        """
        self.name = name
        self.log_interval = log_interval
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.save_to_file = save_to_file
        self.project_name = project_name
        
        self.logger = StructuredLogger(f"SystemMonitor-{name}", debug=debug)
        self.process = psutil.Process()
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Metrics history
        self._metrics_history = []
        self._max_history = 100
        
        # Network monitoring state
        self._last_network_stats = None
        self._network_stats_timestamp = None
        
        # File saving setup
        if self.save_to_file:
            self._setup_file_saving()
    
    def _setup_file_saving(self) -> None:
        """Set up file saving directory and CSV file."""
        # Create directory structure: ./examples/project_name/timestamp/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join("examples", self.project_name, timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create CSV file for metrics
        self.csv_file = os.path.join(self.save_dir, "system_monitor.csv")
        self._write_csv_header()
        
        # Create metadata file
        self.metadata_file = os.path.join(self.save_dir, "system_monitor_metadata.json")
        self._write_metadata()
        
        self.logger.info(
            "File saving initialized",
            save_dir=self.save_dir,
            csv_file=self.csv_file,
            metadata_file=self.metadata_file
        )
    
    def _write_csv_header(self) -> None:
        """Write CSV header to the metrics file."""
        try:
            with open(self.csv_file, 'w', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'datetime', 'memory_percent', 'memory_available_gb',
                    'memory_used_gb', 'cpu_percent', 'disk_usage_percent',
                    'process_memory_mb', 'process_cpu_percent',
                    'network_bytes_sent', 'network_bytes_recv',
                    'network_packets_sent', 'network_packets_recv'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
        except Exception as e:
            self.logger.error("Failed to write CSV header", error=str(e))
    
    def _write_metadata(self) -> None:
        """Write metadata about the monitoring session."""
        try:
            metadata = {
                'monitor_name': self.name,
                'project_name': self.project_name,
                'start_time': datetime.now().isoformat(),
                'log_interval': self.log_interval,
                'memory_threshold': self.memory_threshold,
                'cpu_threshold': self.cpu_threshold,
                'max_history': self._max_history,
                'system_info': {
                    'platform': psutil.sys.platform,
                    'cpu_count': psutil.cpu_count(),
                    'total_memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
                }
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error("Failed to write metadata", error=str(e))
    
    def _save_metrics_to_file(self, metrics: SystemMetrics) -> None:
        """Save a single metrics entry to the CSV file."""
        if not self.save_to_file:
            return
        
        try:
            with open(self.csv_file, 'a', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'datetime', 'memory_percent', 'memory_available_gb',
                    'memory_used_gb', 'cpu_percent', 'disk_usage_percent',
                    'process_memory_mb', 'process_cpu_percent',
                    'network_bytes_sent', 'network_bytes_recv',
                    'network_packets_sent', 'network_packets_recv'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Convert timestamp to datetime string
                dt = datetime.fromtimestamp(metrics.timestamp)
                
                row = {
                    'timestamp': metrics.timestamp,
                    'datetime': dt.isoformat(),
                    'memory_percent': metrics.memory_percent,
                    'memory_available_gb': metrics.memory_available_gb,
                    'memory_used_gb': metrics.memory_used_gb,
                    'cpu_percent': metrics.cpu_percent,
                    'disk_usage_percent': metrics.disk_usage_percent,
                    'process_memory_mb': metrics.process_memory_mb,
                    'process_cpu_percent': metrics.process_cpu_percent,
                    'network_bytes_sent': metrics.network_bytes_sent,
                    'network_bytes_recv': metrics.network_bytes_recv,
                    'network_packets_sent': metrics.network_packets_sent,
                    'network_packets_recv': metrics.network_packets_recv
                }
                writer.writerow(row)
        except Exception as e:
            self.logger.error("Failed to save metrics to file", error=str(e))
    
    def get_current_metrics(self) -> SystemMetrics:
        """
        Get current system metrics.
        
        Returns:
            SystemMetrics object with current system state
        """
        try:
            # System-wide metrics
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage('/')
            
            # Process-specific metrics
            process_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
            process_cpu = self.process.cpu_percent()
            
            # Network bandwidth metrics
            network_bandwidth = self._calculate_network_bandwidth()
            
            metrics = SystemMetrics(
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                memory_used_gb=memory.used / (1024**3),
                cpu_percent=cpu_percent,
                disk_usage_percent=disk.percent,
                process_memory_mb=process_memory,
                process_cpu_percent=process_cpu,
                network_bytes_sent=network_bandwidth['bytes_sent'],
                network_bytes_recv=network_bandwidth['bytes_recv'],
                network_packets_sent=network_bandwidth['packets_sent'],
                network_packets_recv=network_bandwidth['packets_recv'],
                timestamp=time.time()
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error("Failed to collect system metrics", error=str(e))
            # Return default metrics in case of error
            return SystemMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, time.time())
    
    def log_metrics(self, metrics: Optional[SystemMetrics] = None) -> None:
        """
        Log current system metrics with structured logging.
        
        Args:
            metrics: Optional pre-collected metrics. If None, collects current metrics.
        """
        if metrics is None:
            metrics = self.get_current_metrics()
        
        # Check thresholds and log warnings if needed
        warning_context = {}
        if metrics.memory_percent > self.memory_threshold:
            warning_context['memory_warning'] = f"Memory usage {metrics.memory_percent:.1f}% exceeds threshold {self.memory_threshold}%"
        
        if metrics.cpu_percent > self.cpu_threshold:
            warning_context['cpu_warning'] = f"CPU usage {metrics.cpu_percent:.1f}% exceeds threshold {self.cpu_threshold}%"
        
        log_level = "warning" if warning_context else "info"
        log_method = getattr(self.logger, log_level)
        
        log_method(
            f"System metrics for {self.name} - Memory: {metrics.memory_percent:.1f}%, CPU: {metrics.cpu_percent:.1f}%, Process: {metrics.process_memory_mb:.1f}MB",
            memory_percent=metrics.memory_percent,
            memory_available_gb=round(metrics.memory_available_gb, 2),
            memory_used_gb=round(metrics.memory_used_gb, 2),
            cpu_percent=metrics.cpu_percent,
            disk_usage_percent=metrics.disk_usage_percent,
            process_memory_mb=round(metrics.process_memory_mb, 2),
            process_cpu_percent=metrics.process_cpu_percent,
            network_bytes_sent=round(metrics.network_bytes_sent, 2),
            network_bytes_recv=round(metrics.network_bytes_recv, 2),
            network_packets_sent=round(metrics.network_packets_sent, 2),
            network_packets_recv=round(metrics.network_packets_recv, 2),
            **warning_context
        )
        
        # Store in history
        with self._lock:
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self._max_history:
                self._metrics_history.pop(0)
            
            # Save to file
            self._save_metrics_to_file(metrics)
    
    def start_monitoring(self) -> None:
        """Start automatic system monitoring in a background thread."""
        if self._monitoring:
            self.logger.warning("System monitoring already started")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"SystemMonitor-{self.name}",
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("System monitoring started", 
                        log_interval=self.log_interval,
                        memory_threshold=self.memory_threshold,
                        cpu_threshold=self.cpu_threshold)
    
    def stop_monitoring(self) -> None:
        """Stop automatic system monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("System monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self.log_metrics()
                time.sleep(self.log_interval)
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))
                time.sleep(self.log_interval)  # Continue monitoring despite errors
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics from metrics history.
        
        Returns:
            Dictionary with min, max, and average metrics
        """
        with self._lock:
            if not self._metrics_history:
                return {}
            
            memory_percents = [m.memory_percent for m in self._metrics_history]
            cpu_percents = [m.cpu_percent for m in self._metrics_history]
            process_memory = [m.process_memory_mb for m in self._metrics_history]
            
            return {
                'sample_count': len(self._metrics_history),
                'memory_percent': {
                    'min': min(memory_percents),
                    'max': max(memory_percents),
                    'avg': sum(memory_percents) / len(memory_percents)
                },
                'cpu_percent': {
                    'min': min(cpu_percents),
                    'max': max(cpu_percents),
                    'avg': sum(cpu_percents) / len(cpu_percents)
                },
                'process_memory_mb': {
                    'min': min(process_memory),
                    'max': max(process_memory),
                    'avg': sum(process_memory) / len(process_memory)
                }
            }
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
    
    def get_save_directory(self) -> Optional[str]:
        """
        Get the directory where metrics are being saved.
        
        Returns:
            Path to save directory, or None if file saving is disabled
        """
        return self.save_dir if self.save_to_file else None
    
    @staticmethod
    def load_metrics_from_csv(csv_file_path: str) -> List[SystemMetrics]:
        """
        Load system metrics from a CSV file.
        
        Args:
            csv_file_path: Path to the CSV file containing metrics
            
        Returns:
            List of SystemMetrics objects
        """
        metrics_list = []
        
        try:
            with open(csv_file_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    # Check if network columns exist in the CSV
                    has_network_data = 'network_bytes_sent' in row
                    
                    if has_network_data:
                        metrics = SystemMetrics(
                            memory_percent=float(row['memory_percent']),
                            memory_available_gb=float(row['memory_available_gb']),
                            memory_used_gb=float(row['memory_used_gb']),
                            cpu_percent=float(row['cpu_percent']),
                            disk_usage_percent=float(row['disk_usage_percent']),
                            process_memory_mb=float(row['process_memory_mb']),
                            process_cpu_percent=float(row['process_cpu_percent']),
                            network_bytes_sent=float(row['network_bytes_sent']),
                            network_bytes_recv=float(row['network_bytes_recv']),
                            network_packets_sent=float(row['network_packets_sent']),
                            network_packets_recv=float(row['network_packets_recv']),
                            timestamp=float(row['timestamp'])
                        )
                    else:
                        # Create metrics without network data for older CSV files
                        metrics = SystemMetrics(
                            memory_percent=float(row['memory_percent']),
                            memory_available_gb=float(row['memory_available_gb']),
                            memory_used_gb=float(row['memory_used_gb']),
                            cpu_percent=float(row['cpu_percent']),
                            disk_usage_percent=float(row['disk_usage_percent']),
                            process_memory_mb=float(row['process_memory_mb']),
                            process_cpu_percent=float(row['process_cpu_percent']),
                            network_bytes_sent=0,
                            network_bytes_recv=0,
                            network_packets_sent=0,
                            network_packets_recv=0,
                            timestamp=float(row['timestamp'])
                        )
                    
                    metrics_list.append(metrics)
            
            return metrics_list
            
        except Exception as e:
            print(f"Error loading metrics from {csv_file_path}: {e}")
            return []
    
    @staticmethod
    def load_metadata_from_json(metadata_file_path: str) -> Dict[str, Any]:
        """
        Load metadata from a JSON file.
        
        Args:
            metadata_file_path: Path to the metadata JSON file
            
        Returns:
            Dictionary containing metadata
        """
        try:
            with open(metadata_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata from {metadata_file_path}: {e}")
            return {}

    def _get_network_stats(self) -> Dict[str, float]:
        """
        Get current network statistics for the process.
        
        Returns:
            Dictionary with network statistics
        """
        try:
            # Get network connections for the current process
            connections = self.process.connections()
            
            # Get system-wide network stats
            net_io = psutil.net_io_counters()
            
            # Calculate process-specific network usage
            process_net_io = {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0}
            
            # For now, we'll use system-wide stats since process-specific network I/O
            # is not directly available in psutil. In a real implementation, you might
            # want to use more sophisticated methods like eBPF or system calls.
            process_net_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            return process_net_io
            
        except Exception as e:
            self.logger.error("Failed to get network stats", error=str(e))
            return {'bytes_sent': 0, 'bytes_recv': 0, 'packets_sent': 0, 'packets_recv': 0}
    
    def _calculate_network_bandwidth(self) -> Dict[str, float]:
        """
        Calculate network bandwidth usage since last measurement.
        
        Returns:
            Dictionary with bandwidth metrics
        """
        current_stats = self._get_network_stats()
        current_time = time.time()
        
        if self._last_network_stats is None:
            # First measurement, store initial values
            self._last_network_stats = current_stats
            self._network_stats_timestamp = current_time
            return {
                'bytes_sent': 0,
                'bytes_recv': 0,
                'packets_sent': 0,
                'packets_recv': 0
            }
        
        # Calculate differences
        time_diff = current_time - self._network_stats_timestamp
        if time_diff <= 0:
            return {
                'bytes_sent': 0,
                'bytes_recv': 0,
                'packets_sent': 0,
                'packets_recv': 0
            }
        
        bytes_sent_diff = current_stats['bytes_sent'] - self._last_network_stats['bytes_sent']
        bytes_recv_diff = current_stats['bytes_recv'] - self._last_network_stats['bytes_recv']
        packets_sent_diff = current_stats['packets_sent'] - self._last_network_stats['packets_sent']
        packets_recv_diff = current_stats['packets_recv'] - self._last_network_stats['packets_recv']
        
        # Update stored values
        self._last_network_stats = current_stats
        self._network_stats_timestamp = current_time
        
        return {
            'bytes_sent': bytes_sent_diff,
            'bytes_recv': bytes_recv_diff,
            'packets_sent': packets_sent_diff,
            'packets_recv': packets_recv_diff
        }

# Convenience functions for quick monitoring
def log_memory_usage(logger: StructuredLogger, context: str = "") -> None:
    """
    Log current memory usage with a structured logger.
    
    Args:
        logger: StructuredLogger instance
        context: Additional context for the log message
    """
    try:
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        logger.info(f"Memory usage {context}",
                   system_memory_percent=memory.percent,
                   system_memory_available_gb=round(memory.available / (1024**3), 2),
                   process_memory_mb=round(process_memory, 2))
    except Exception as e:
        logger.error("Failed to log memory usage", error=str(e))

def check_memory_threshold(threshold_percent: float = 85.0) -> bool:
    """
    Check if system memory usage exceeds threshold.
    
    Args:
        threshold_percent: Memory usage threshold percentage
        
    Returns:
        True if memory usage exceeds threshold
    """
    try:
        memory = psutil.virtual_memory()
        return memory.percent > threshold_percent
    except Exception:
        return False  # Assume safe if we can't check