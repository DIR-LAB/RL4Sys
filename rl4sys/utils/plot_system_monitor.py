#!/usr/bin/env python3
"""
System Monitor Data Plotting Script

This script loads system monitor data from CSV files and creates
comprehensive plots showing system performance over time.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import csv

# Add the parent directory to the Python path to find rl4sys module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # examples/
project_root = os.path.dirname(parent_dir)  # rl4sys/
sys.path.insert(0, project_root)

from rl4sys.utils.system_monitor import SystemMonitor, SystemMetrics

class SystemMonitorPlotter:
    """
    Plotting utility for system monitor data.
    
    Provides comprehensive visualization of system performance metrics
    including memory usage, CPU usage, and process-specific metrics.
    """
    
    def __init__(self, csv_file_path: str, metadata_file_path: Optional[str] = None):
        """
        Initialize the plotter with data files.
        
        Args:
            csv_file_path: Path to the CSV file containing metrics
            metadata_file_path: Optional path to metadata JSON file
        """
        self.csv_file_path = csv_file_path
        self.metadata_file_path = metadata_file_path
        
        # Load data
        self.metrics = SystemMonitor.load_metrics_from_csv(csv_file_path)
        self.metadata = {}
        if metadata_file_path and os.path.exists(metadata_file_path):
            self.metadata = SystemMonitor.load_metadata_from_json(metadata_file_path)
        
        if not self.metrics:
            raise ValueError(f"No metrics data found in {csv_file_path}")
        
        # Convert to pandas DataFrame for easier plotting
        self.df = self._convert_to_dataframe()
        
        print(f"Loaded {len(self.metrics)} metrics samples")
        if self.metadata:
            print(f"Monitoring session: {self.metadata.get('monitor_name', 'Unknown')}")
            print(f"Project: {self.metadata.get('project_name', 'Unknown')}")
    
    def _convert_to_dataframe(self) -> pd.DataFrame:
        """Convert metrics list to pandas DataFrame."""
        data = []
        for metric in self.metrics:
            data.append({
                'timestamp': metric.timestamp,
                'datetime': datetime.fromtimestamp(metric.timestamp),
                'memory_percent': metric.memory_percent,
                'memory_available_gb': metric.memory_available_gb,
                'memory_used_gb': metric.memory_used_gb,
                'cpu_percent': metric.cpu_percent,
                'disk_usage_percent': metric.disk_usage_percent,
                'process_memory_mb': metric.process_memory_mb,
                'process_cpu_percent': metric.process_cpu_percent
            })
            
            # Add network columns if they exist in the metric
            if hasattr(metric, 'network_bytes_sent'):
                data[-1].update({
                    'network_bytes_sent': metric.network_bytes_sent,
                    'network_bytes_recv': metric.network_bytes_recv,
                    'network_packets_sent': metric.network_packets_sent,
                    'network_packets_recv': metric.network_packets_recv
                })
            else:
                # Add default values for older data without network metrics
                data[-1].update({
                    'network_bytes_sent': 0,
                    'network_bytes_recv': 0,
                    'network_packets_sent': 0,
                    'network_packets_recv': 0
                })
        
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        return df
    
    def plot_overview(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Create an overview plot with all key metrics.
        
        Args:
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(4, 2, figsize=(15, 16))
        fig.suptitle('System Monitor Overview', fontsize=16, fontweight='bold')
        
        # Memory usage
        axes[0, 0].plot(self.df.index, self.df['memory_percent'], 'b-', linewidth=2)
        axes[0, 0].set_title('System Memory Usage (%)')
        axes[0, 0].set_ylabel('Memory (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 100)
        
        # CPU usage
        axes[0, 1].plot(self.df.index, self.df['cpu_percent'], 'r-', linewidth=2)
        axes[0, 1].set_title('System CPU Usage (%)')
        axes[0, 1].set_ylabel('CPU (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 100)
        
        # Process memory
        axes[1, 0].plot(self.df.index, self.df['process_memory_mb'], 'g-', linewidth=2)
        axes[1, 0].set_title('Process Memory Usage (MB)')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Process CPU
        axes[1, 1].plot(self.df.index, self.df['process_cpu_percent'], 'orange', linewidth=2)
        axes[1, 1].set_title('Process CPU Usage (%)')
        axes[1, 1].set_ylabel('CPU (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 100)
        
        # Memory breakdown
        axes[2, 0].plot(self.df.index, self.df['memory_used_gb'], 'purple', linewidth=2, label='Used')
        axes[2, 0].plot(self.df.index, self.df['memory_available_gb'], 'lightblue', linewidth=2, label='Available')
        axes[2, 0].set_title('Memory Breakdown (GB)')
        axes[2, 0].set_ylabel('Memory (GB)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Disk usage
        axes[2, 1].plot(self.df.index, self.df['disk_usage_percent'], 'brown', linewidth=2)
        axes[2, 1].set_title('Disk Usage (%)')
        axes[2, 1].set_ylabel('Disk (%)')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_ylim(0, 100)
        
        # Network bandwidth - bytes sent/received
        axes[3, 0].plot(self.df.index, self.df['network_bytes_sent'], 'blue', linewidth=2, label='Bytes Sent')
        axes[3, 0].plot(self.df.index, self.df['network_bytes_recv'], 'green', linewidth=2, label='Bytes Received')
        axes[3, 0].set_title('Network Bandwidth (Bytes)')
        axes[3, 0].set_ylabel('Bytes')
        axes[3, 0].legend()
        axes[3, 0].grid(True, alpha=0.3)
        
        # Network packets
        axes[3, 1].plot(self.df.index, self.df['network_packets_sent'], 'red', linewidth=2, label='Packets Sent')
        axes[3, 1].plot(self.df.index, self.df['network_packets_recv'], 'orange', linewidth=2, label='Packets Received')
        axes[3, 1].set_title('Network Packets')
        axes[3, 1].set_ylabel('Packets')
        axes[3, 1].legend()
        axes[3, 1].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, len(self.df) // 10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Overview plot saved to {save_path}")
        
        if show:
            plt.show()
    
    def plot_memory_analysis(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Create detailed memory analysis plots.
        
        Args:
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Memory Usage Analysis', fontsize=16, fontweight='bold')
        
        # System memory percentage with thresholds
        axes[0, 0].plot(self.df.index, self.df['memory_percent'], 'b-', linewidth=2, label='Memory Usage')
        if self.metadata and 'memory_threshold' in self.metadata:
            threshold = self.metadata['memory_threshold']
            axes[0, 0].axhline(y=threshold, color='r', linestyle='--', alpha=0.7, 
                              label=f'Threshold ({threshold}%)')
        axes[0, 0].set_title('System Memory Usage')
        axes[0, 0].set_ylabel('Memory (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 100)
        
        # Memory usage vs available
        axes[0, 1].plot(self.df.index, self.df['memory_used_gb'], 'purple', linewidth=2, label='Used')
        axes[0, 1].plot(self.df.index, self.df['memory_available_gb'], 'lightblue', linewidth=2, label='Available')
        axes[0, 1].set_title('Memory Breakdown')
        axes[0, 1].set_ylabel('Memory (GB)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Process memory usage
        axes[1, 0].plot(self.df.index, self.df['process_memory_mb'], 'g-', linewidth=2)
        axes[1, 0].set_title('Process Memory Usage')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Memory usage histogram
        axes[1, 1].hist(self.df['memory_percent'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].set_title('Memory Usage Distribution')
        axes[1, 1].set_xlabel('Memory (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes.flat:
            if hasattr(ax, 'xaxis'):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, len(self.df) // 10)))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Memory analysis plot saved to {save_path}")
        
        if show:
            plt.show()
    
    def plot_cpu_analysis(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Create detailed CPU analysis plots.
        
        Args:
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CPU Usage Analysis', fontsize=16, fontweight='bold')
        
        # System CPU usage with thresholds
        axes[0, 0].plot(self.df.index, self.df['cpu_percent'], 'r-', linewidth=2, label='CPU Usage')
        if self.metadata and 'cpu_threshold' in self.metadata:
            threshold = self.metadata['cpu_threshold']
            axes[0, 0].axhline(y=threshold, color='r', linestyle='--', alpha=0.7, 
                              label=f'Threshold ({threshold}%)')
        axes[0, 0].set_title('System CPU Usage')
        axes[0, 0].set_ylabel('CPU (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 100)
        
        # Process CPU usage
        axes[0, 1].plot(self.df.index, self.df['process_cpu_percent'], 'orange', linewidth=2)
        axes[0, 1].set_title('Process CPU Usage')
        axes[0, 1].set_ylabel('CPU (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 100)
        
        # System vs Process CPU comparison
        axes[1, 0].plot(self.df.index, self.df['cpu_percent'], 'r-', linewidth=2, label='System CPU')
        axes[1, 0].plot(self.df.index, self.df['process_cpu_percent'], 'orange', linewidth=2, label='Process CPU')
        axes[1, 0].set_title('System vs Process CPU Usage')
        axes[1, 0].set_ylabel('CPU (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 100)
        
        # CPU usage histogram
        axes[1, 1].hist(self.df['cpu_percent'], bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_title('CPU Usage Distribution')
        axes[1, 1].set_xlabel('CPU (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes.flat:
            if hasattr(ax, 'xaxis'):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, len(self.df) // 10)))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CPU analysis plot saved to {save_path}")
        
        if show:
            plt.show()
    
    def plot_network_analysis(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Create detailed network analysis plots.
        
        Args:
            save_path: Optional path to save the plot
            show: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Network Bandwidth Analysis', fontsize=16, fontweight='bold')
        
        # Network bandwidth over time
        axes[0, 0].plot(self.df.index, self.df['network_bytes_sent'], 'blue', linewidth=2, label='Bytes Sent')
        axes[0, 0].plot(self.df.index, self.df['network_bytes_recv'], 'green', linewidth=2, label='Bytes Received')
        axes[0, 0].set_title('Network Bandwidth Usage')
        axes[0, 0].set_ylabel('Bytes')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Network packets over time
        axes[0, 1].plot(self.df.index, self.df['network_packets_sent'], 'red', linewidth=2, label='Packets Sent')
        axes[0, 1].plot(self.df.index, self.df['network_packets_recv'], 'orange', linewidth=2, label='Packets Received')
        axes[0, 1].set_title('Network Packet Count')
        axes[0, 1].set_ylabel('Packets')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bandwidth distribution histogram
        axes[1, 0].hist(self.df['network_bytes_sent'], bins=20, alpha=0.7, color='blue', edgecolor='black', label='Bytes Sent')
        axes[1, 0].hist(self.df['network_bytes_recv'], bins=20, alpha=0.7, color='green', edgecolor='black', label='Bytes Received')
        axes[1, 0].set_title('Network Bandwidth Distribution')
        axes[1, 0].set_xlabel('Bytes')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Total network activity
        total_sent = self.df['network_bytes_sent'].sum()
        total_recv = self.df['network_bytes_recv'].sum()
        total_packets_sent = self.df['network_packets_sent'].sum()
        total_packets_recv = self.df['network_packets_recv'].sum()
        
        # Create a summary text box
        summary_text = f"""Network Summary:
Total Bytes Sent: {total_sent:,.0f}
Total Bytes Received: {total_recv:,.0f}
Total Packets Sent: {total_packets_sent:,.0f}
Total Packets Received: {total_packets_recv:,.0f}
Avg Bytes/Sample: {self.df['network_bytes_sent'].mean():.1f} sent, {self.df['network_bytes_recv'].mean():.1f} recv
Max Bytes/Sample: {self.df['network_bytes_sent'].max():.1f} sent, {self.df['network_bytes_recv'].max():.1f} recv"""
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        axes[1, 1].set_title('Network Statistics Summary')
        axes[1, 1].axis('off')
        
        # Format x-axis for time-based plots
        for ax in [axes[0, 0], axes[0, 1]]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, len(self.df) // 10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Network analysis plot saved to {save_path}")
        
        if show:
            plt.show()
    
    def print_summary_statistics(self) -> None:
        """Print summary statistics of the monitoring data."""
        print("\n" + "="*60)
        print("SYSTEM MONITOR SUMMARY STATISTICS")
        print("="*60)
        
        # Time range
        start_time = self.df.index[0]
        end_time = self.df.index[-1]
        duration = end_time - start_time
        
        print(f"Monitoring Duration: {duration}")
        print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Samples: {len(self.df)}")
        
        # Memory statistics
        print(f"\nMEMORY USAGE:")
        print(f"  System Memory (%):")
        print(f"    Average: {self.df['memory_percent'].mean():.2f}%")
        print(f"    Maximum: {self.df['memory_percent'].max():.2f}%")
        print(f"    Minimum: {self.df['memory_percent'].min():.2f}%")
        print(f"    Std Dev: {self.df['memory_percent'].std():.2f}%")
        
        print(f"  Process Memory (MB):")
        print(f"    Average: {self.df['process_memory_mb'].mean():.2f} MB")
        print(f"    Maximum: {self.df['process_memory_mb'].max():.2f} MB")
        print(f"    Minimum: {self.df['process_memory_mb'].min():.2f} MB")
        
        # CPU statistics
        print(f"\nCPU USAGE:")
        print(f"  System CPU (%):")
        print(f"    Average: {self.df['cpu_percent'].mean():.2f}%")
        print(f"    Maximum: {self.df['cpu_percent'].max():.2f}%")
        print(f"    Minimum: {self.df['cpu_percent'].min():.2f}%")
        print(f"    Std Dev: {self.df['cpu_percent'].std():.2f}%")
        
        print(f"  Process CPU (%):")
        print(f"    Average: {self.df['process_cpu_percent'].mean():.2f}%")
        print(f"    Maximum: {self.df['process_cpu_percent'].max():.2f}%")
        print(f"    Minimum: {self.df['process_cpu_percent'].min():.2f}%")
        
        # Network statistics
        print(f"\nNETWORK USAGE:")
        total_sent = self.df['network_bytes_sent'].sum()
        total_recv = self.df['network_bytes_recv'].sum()
        
        if total_sent > 0 or total_recv > 0:
            print(f"  Network Bytes Sent:")
            print(f"    Total: {total_sent:,.0f} bytes")
            print(f"    Average: {self.df['network_bytes_sent'].mean():.2f} bytes/sample")
            print(f"    Maximum: {self.df['network_bytes_sent'].max():.2f} bytes/sample")
            
            print(f"  Network Bytes Received:")
            print(f"    Total: {total_recv:,.0f} bytes")
            print(f"    Average: {self.df['network_bytes_recv'].mean():.2f} bytes/sample")
            print(f"    Maximum: {self.df['network_bytes_recv'].max():.2f} bytes/sample")
            
            print(f"  Network Packets:")
            print(f"    Total Sent: {self.df['network_packets_sent'].sum():,.0f}")
            print(f"    Total Received: {self.df['network_packets_recv'].sum():,.0f}")
        else:
            print(f"  Network monitoring data not available (older data format)")
        
        # Threshold violations
        if self.metadata:
            memory_threshold = self.metadata.get('memory_threshold', 85.0)
            cpu_threshold = self.metadata.get('cpu_threshold', 90.0)
            
            memory_violations = (self.df['memory_percent'] > memory_threshold).sum()
            cpu_violations = (self.df['cpu_percent'] > cpu_threshold).sum()
            
            print(f"\nTHRESHOLD VIOLATIONS:")
            print(f"  Memory > {memory_threshold}%: {memory_violations} times ({memory_violations/len(self.df)*100:.1f}%)")
            print(f"  CPU > {cpu_threshold}%: {cpu_violations} times ({cpu_violations/len(self.df)*100:.1f}%)")
        
        print("="*60)

def main():
    """Main function to run the plotting script."""
    parser = argparse.ArgumentParser(description='Plot system monitor data')
    parser.add_argument('--csv-file', default='/home/yomi/0Projects/RL4Sys/examples/rl4sys_server/20250709_042552/system_monitor.csv', help='Path to the CSV file containing metrics')
    parser.add_argument('--metadata', default='/home/yomi/0Projects/RL4Sys/examples/rl4sys_server/20250709_042552/system_monitor_metadata.json', help='Path to the metadata JSON file')
    parser.add_argument('--output-dir', default='/home/yomi/0Projects/RL4Sys/examples/rl4sys_server/20250709_042552', help='Directory to save plots')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    parser.add_argument('--plot-type', choices=['overview', 'memory', 'cpu', 'network', 'all'], 
                       default='all', help='Type of plots to generate')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found")
        sys.exit(1)
    
    try:
        # Create plotter
        plotter = SystemMonitorPlotter(args.csv_file, args.metadata)
        
        # Print summary statistics
        plotter.print_summary_statistics()
        
        # Create output directory if specified
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # Generate plots based on type
        show_plots = not args.no_show
        
        if args.plot_type in ['overview', 'all']:
            output_path = os.path.join(args.output_dir, 'system_overview.png') if args.output_dir else None
            plotter.plot_overview(save_path=output_path, show=show_plots)
        
        if args.plot_type in ['memory', 'all']:
            output_path = os.path.join(args.output_dir, 'memory_analysis.png') if args.output_dir else None
            plotter.plot_memory_analysis(save_path=output_path, show=show_plots)
        
        if args.plot_type in ['cpu', 'all']:
            output_path = os.path.join(args.output_dir, 'cpu_analysis.png') if args.output_dir else None
            plotter.plot_cpu_analysis(save_path=output_path, show=show_plots)
        
        if args.plot_type in ['network', 'all']:
            output_path = os.path.join(args.output_dir, 'network_analysis.png') if args.output_dir else None
            plotter.plot_network_analysis(save_path=output_path, show=show_plots)
        
        print("Plotting completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 