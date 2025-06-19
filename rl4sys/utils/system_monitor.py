"""
System health monitoring utilities for RL4Sys.

This module provides tools for monitoring memory usage, system resources,
and performance metrics across the distributed training framework.
"""

import os
import time
import psutil
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
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
    timestamp: float

class SystemMonitor:
    """
    System health monitoring for RL4Sys components.
    
    Provides memory usage tracking, CPU monitoring, and performance metrics
    with configurable logging intervals and thresholds.
    """
    
    def __init__(self, 
                 name: str = "SystemMonitor",
                 log_interval: float = 30.0,
                 memory_threshold: float = 85.0,
                 cpu_threshold: float = 90.0,
                 debug: bool = False):
        """
        Initialize system monitor.
        
        Args:
            name: Monitor instance name for logging
            log_interval: Seconds between automatic metric logging
            memory_threshold: Memory usage percentage threshold for warnings
            cpu_threshold: CPU usage percentage threshold for warnings
            debug: Enable debug logging
        """
        self.name = name
        self.log_interval = log_interval
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        
        self.logger = StructuredLogger(f"SystemMonitor-{name}", debug=debug)
        self.process = psutil.Process()
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Metrics history
        self._metrics_history = []
        self._max_history = 100
        
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
            
            metrics = SystemMetrics(
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                memory_used_gb=memory.used / (1024**3),
                cpu_percent=cpu_percent,
                disk_usage_percent=disk.percent,
                process_memory_mb=process_memory,
                process_cpu_percent=process_cpu,
                timestamp=time.time()
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error("Failed to collect system metrics", error=str(e))
            # Return default metrics in case of error
            return SystemMetrics(0, 0, 0, 0, 0, 0, 0, time.time())
    
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
            **warning_context
        )
        
        # Store in history
        with self._lock:
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self._max_history:
                self._metrics_history.pop(0)
    
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