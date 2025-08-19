# System Monitor Documentation

This document describes the enhanced System Monitor functionality in RL4Sys, which now includes file saving capabilities and plotting tools.

## Overview

The System Monitor collects system performance metrics including:
- System memory usage (percentage and amounts)
- CPU usage (system and process-specific)
- Disk usage
- Process memory consumption
- Timestamps for all measurements

## Features

### 1. File Saving
- Automatically saves metrics to CSV files
- Creates organized directory structure: `./examples/project_name/timestamp/`
- Saves metadata about the monitoring session
- Thread-safe file writing

### 2. Plotting Tools
- Comprehensive visualization of system metrics
- Multiple plot types: overview, memory analysis, CPU analysis
- Statistical summaries and threshold violation tracking
- High-quality PNG output

## Usage

### Basic System Monitor Usage

```python
from rl4sys.utils.system_monitor import SystemMonitor

# Create monitor with file saving enabled
monitor = SystemMonitor(
    name="MyMonitor",
    log_interval=30.0,  # seconds between measurements
    memory_threshold=85.0,
    cpu_threshold=90.0,
    debug=False,
    save_to_file=True,  # Enable file saving
    project_name="my_project"  # Organizes files by project
)

# Start monitoring
monitor.start_monitoring()

# ... your application code ...

# Stop monitoring
monitor.stop_monitoring()

# Get save directory
save_dir = monitor.get_save_directory()
print(f"Data saved to: {save_dir}")
```

### Test Script

Run the test script to see the system monitor in action:

```bash
# Basic test (60 seconds, 5-second intervals)
python examples/test_system_monitor.py

# Custom parameters
python examples/test_system_monitor.py --duration 120 --interval 10 --project-name "my_test"

# With debug logging
python examples/test_system_monitor.py --debug
```

### Plotting Data

After collecting data, use the plotting script to visualize it:

```bash
# Basic plotting (shows all plot types)
python examples/plot_system_monitor.py path/to/system_monitor.csv

# With metadata file
python examples/plot_system_monitor.py path/to/system_monitor.csv --metadata path/to/system_monitor_metadata.json

# Save plots to directory
python examples/plot_system_monitor.py path/to/system_monitor.csv --output-dir plots/

# Specific plot types
python examples/plot_system_monitor.py path/to/system_monitor.csv --plot-type memory
python examples/plot_system_monitor.py path/to/system_monitor.csv --plot-type cpu
python examples/plot_system_monitor.py path/to/system_monitor.csv --plot-type overview

# Generate plots without displaying them
python examples/plot_system_monitor.py path/to/system_monitor.csv --no-show --output-dir plots/
```

## File Structure

When file saving is enabled, the following structure is created:

```
examples/
└── project_name/
    └── YYYYMMDD_HHMMSS/
        ├── system_monitor.csv          # Metrics data
        └── system_monitor_metadata.json # Session metadata
```

### CSV File Format

The CSV file contains the following columns:
- `timestamp`: Unix timestamp
- `datetime`: Human-readable datetime
- `memory_percent`: System memory usage percentage
- `memory_available_gb`: Available memory in GB
- `memory_used_gb`: Used memory in GB
- `cpu_percent`: System CPU usage percentage
- `disk_usage_percent`: Disk usage percentage
- `process_memory_mb`: Process memory usage in MB
- `process_cpu_percent`: Process CPU usage percentage

### Metadata File

The metadata JSON file contains:
- Monitor configuration (name, thresholds, intervals)
- System information (platform, CPU count, total memory)
- Session start time and duration
- Project information

## Plot Types

### 1. Overview Plot
Shows all key metrics in a 3x2 grid:
- System memory usage (%)
- System CPU usage (%)
- Process memory usage (MB)
- Process CPU usage (%)
- Memory breakdown (used vs available)
- Disk usage (%)

### 2. Memory Analysis
Detailed memory-focused plots:
- System memory usage with thresholds
- Memory breakdown (used vs available)
- Process memory usage
- Memory usage distribution histogram

### 3. CPU Analysis
Detailed CPU-focused plots:
- System CPU usage with thresholds
- Process CPU usage
- System vs Process CPU comparison
- CPU usage distribution histogram

## Integration with RL4Sys Server

The RL4Sys server automatically uses the enhanced system monitor:

```python
# In server.py, the system monitor is initialized with file saving:
self.system_monitor = SystemMonitor(
    "RL4SysServer", 
    debug=debug,
    save_to_file=True,
    project_name="rl4sys_server"
)
```

This means every server run will automatically save system metrics to:
```
examples/rl4sys_server/YYYYMMDD_HHMMSS/
```

## Dependencies

The plotting functionality requires:
- `matplotlib` - For creating plots
- `pandas` - For data manipulation
- `numpy` - For numerical operations

Install with:
```bash
pip install matplotlib pandas numpy
```

## Example Workflow

1. **Start monitoring**:
   ```bash
   python examples/test_system_monitor.py --duration 300 --project-name "training_run"
   ```

2. **Check generated files**:
   ```bash
   ls examples/training_run/
   ```

3. **Plot the data**:
   ```bash
   python examples/plot_system_monitor.py examples/training_run/YYYYMMDD_HHMMSS/system_monitor.csv --metadata examples/training_run/YYYYMMDD_HHMMSS/system_monitor_metadata.json
   ```

4. **Analyze results**:
   - Review the summary statistics printed by the plotting script
   - Examine the generated PNG files
   - Check for threshold violations

## Troubleshooting

### Common Issues

1. **Permission errors**: Ensure write permissions in the examples directory
2. **Missing dependencies**: Install matplotlib, pandas, and numpy
3. **No data in plots**: Check that the CSV file contains data and timestamps are valid
4. **Memory issues**: Reduce monitoring frequency or duration for long runs

### Debug Mode

Enable debug logging to see detailed information:
```python
monitor = SystemMonitor(debug=True, save_to_file=True, project_name="debug_test")
```

## Performance Considerations

- File I/O is buffered and thread-safe
- CSV writing is efficient for large datasets
- Plotting can be memory-intensive for very long monitoring sessions
- Consider using `--no-show` for batch processing of multiple datasets

## Future Enhancements

Potential improvements:
- Real-time plotting during monitoring
- Database storage for long-term analysis
- Alert system for threshold violations
- Integration with external monitoring tools
- Custom metric collection plugins 