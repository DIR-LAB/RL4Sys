#!/usr/bin/env python3
"""
Test script for Network Monitoring functionality.

This script demonstrates the enhanced SystemMonitor with network bandwidth tracking.
"""

import os
import sys
import time
import argparse
import socket
import threading
from datetime import datetime

# Add the parent directory to the Python path to find rl4sys module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # examples/
project_root = os.path.dirname(parent_dir)  # rl4sys/
sys.path.insert(0, project_root)

from rl4sys.utils.system_monitor import SystemMonitor
from rl4sys.utils.logging_config import RL4SysLogConfig

def simulate_network_activity(duration: int):
    """
    Simulate network activity by creating socket connections.
    
    Args:
        duration: Duration in seconds to simulate network activity
    """
    print(f"Simulating network activity for {duration} seconds...")
    
    # Create a simple HTTP request simulation
    def make_http_request():
        try:
            # Create a socket connection (this will generate network activity)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            
            # Try to connect to a well-known service (this will fail but generate network activity)
            sock.connect_ex(('8.8.8.8', 80))
            sock.close()
        except:
            pass  # Expected to fail, but generates network activity
    
    start_time = time.time()
    while time.time() - start_time < duration:
        # Make multiple requests to generate network activity
        for _ in range(5):
            make_http_request()
        time.sleep(0.1)  # Small delay between requests

def main():
    """Main function to test network monitoring."""
    parser = argparse.ArgumentParser(description='Test network monitoring with file saving')
    parser.add_argument('--duration', type=int, default=60, help='Monitoring duration in seconds')
    parser.add_argument('--interval', type=float, default=2.0, help='Monitoring interval in seconds')
    parser.add_argument('--project-name', type=str, default='network_test', help='Project name for file organization')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--network-simulation', action='store_true', help='Simulate network activity')
    
    args = parser.parse_args()
    
    # Set up logging
    config = RL4SysLogConfig.get_default_config(
        log_level="DEBUG" if args.debug else "INFO",
        structured_logging=True
    )
    RL4SysLogConfig.setup_logging(config_dict=config)
    
    print(f"Starting network monitoring for {args.duration} seconds...")
    print(f"Monitoring interval: {args.interval} seconds")
    print(f"Project name: {args.project_name}")
    print(f"Debug mode: {args.debug}")
    print(f"Network simulation: {args.network_simulation}")
    
    # Create system monitor with file saving enabled
    monitor = SystemMonitor(
        name="NetworkTestMonitor",
        log_interval=args.interval,
        memory_threshold=80.0,
        cpu_threshold=85.0,
        debug=args.debug,
        save_to_file=True,
        project_name=args.project_name
    )
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Start network simulation in a separate thread if requested
        network_thread = None
        if args.network_simulation:
            network_thread = threading.Thread(
                target=simulate_network_activity,
                args=(args.duration,),
                daemon=True
            )
            network_thread.start()
            print("Network simulation thread started")
        
        # Main monitoring loop
        print("Monitoring system metrics...")
        start_time = time.time()
        
        while time.time() - start_time < args.duration:
            # Simulate some CPU work to generate interesting metrics
            for _ in range(100000):
                _ = 1 + 1
            
            # Sleep a bit
            time.sleep(1)
            
            # Print progress
            elapsed = time.time() - start_time
            remaining = args.duration - elapsed
            print(f"Progress: {elapsed:.1f}s / {args.duration}s (remaining: {remaining:.1f}s)")
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Wait for network thread to finish
        if network_thread and network_thread.is_alive():
            network_thread.join(timeout=5.0)
        
        # Get save directory
        save_dir = monitor.get_save_directory()
        if save_dir:
            print(f"\nMonitoring data saved to: {save_dir}")
            print(f"CSV file: {os.path.join(save_dir, 'system_monitor.csv')}")
            print(f"Metadata file: {os.path.join(save_dir, 'system_monitor_metadata.json')}")
        
        # Print summary statistics
        summary = monitor.get_metrics_summary()
        if summary:
            print("\nSummary Statistics:")
            print(f"  Sample count: {summary.get('sample_count', 0)}")
            print(f"  Memory usage - Avg: {summary.get('memory_percent', {}).get('avg', 0):.2f}%")
            print(f"  CPU usage - Avg: {summary.get('cpu_percent', {}).get('avg', 0):.2f}%")
            print(f"  Process memory - Avg: {summary.get('process_memory_mb', {}).get('avg', 0):.2f} MB")
        
        print("\nNetwork monitoring completed successfully!")
        
        # Instructions for plotting
        if save_dir:
            csv_file = os.path.join(save_dir, 'system_monitor.csv')
            metadata_file = os.path.join(save_dir, 'system_monitor_metadata.json')
            
            print(f"\nTo plot the data, run:")
            print(f"python examples/plot_system_monitor.py --csv-file {csv_file} --metadata {metadata_file}")
            print(f"\nFor network-specific analysis:")
            print(f"python examples/plot_system_monitor.py --csv-file {csv_file} --metadata {metadata_file} --plot-type network")
        
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        monitor.stop_monitoring()
    except Exception as e:
        print(f"Error during monitoring: {e}")
        monitor.stop_monitoring()
        sys.exit(1)

if __name__ == '__main__':
    main() 