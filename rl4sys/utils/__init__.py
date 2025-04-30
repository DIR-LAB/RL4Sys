"""
RL4Sys Utils Package

This package contains utility functions and classes for configuration management,
logging, plotting, and data serialization/deserialization.
"""

# Configuration management
from .conf_loader import ConfigLoader

# Logging utilities
from .logger import (
    Logger,
    EpochLogger,
    setup_logger_kwargs,
    colorize,
    convert_json,
    is_json_serializable,
    statistics_scalar
)

# Plotting utilities
from .plot import (
    plot_data,
    get_datasets,
    get_all_datasets,
    make_plots,
    get_newest_dataset,
    get_simple_dataset_plot
)

# Serialization utilities
from .util import (
    serialize_tensor,
    deserialize_tensor,
    serialize_action,
    deserialize_action,
    serialize_model,
    deserialize_model
)

__all__ = [
    # Configuration
    'ConfigLoader',
    
    # Logging
    'Logger',
    'EpochLogger',
    'setup_logger_kwargs',
    'colorize',
    'convert_json',
    'is_json_serializable',
    'statistics_scalar',
    
    # Plotting
    'plot_data',
    'get_datasets',
    'get_all_datasets',
    'make_plots',
    'get_newest_dataset',
    'get_simple_dataset_plot',
    
    # Serialization
    'serialize_tensor',
    'deserialize_tensor',
    'serialize_action',
    'deserialize_action',
    'serialize_model',
    'deserialize_model'
]
