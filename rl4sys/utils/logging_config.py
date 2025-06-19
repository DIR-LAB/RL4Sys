"""
Centralized logging configuration for RL4Sys.

This module provides utilities for setting up consistent logging across
the entire RL4Sys framework, including performance-optimized logging
for training loops and unified configuration management.
"""

import os
import logging
import logging.config
from typing import Dict, Any, Optional
from enum import Enum

class LogLevel(Enum):
    """Supported logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class RL4SysLogConfig:
    """Centralized logging configuration manager."""
    
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    STRUCTURED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(context)s'
    
    @staticmethod
    def get_default_config(
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        enable_tensorboard: bool = True,
        structured_logging: bool = False
    ) -> Dict[str, Any]:
        """
        Get default logging configuration dictionary.
        
        Args:
            log_level: Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path. If None, only console logging is used.
            enable_tensorboard: Whether to enable TensorBoard integration
            structured_logging: Whether to use structured logging format
            
        Returns:
            Dictionary suitable for logging.config.dictConfig()
        """
        
        handlers = {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'structured' if structured_logging else 'standard',
                'stream': 'ext://sys.stdout'
            }
        }
        
        formatters = {
            'standard': {
                'format': RL4SysLogConfig.DEFAULT_FORMAT,
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'structured': {
                'format': RL4SysLogConfig.STRUCTURED_FORMAT,
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        }
        
        if log_file:
            handlers['file'] = {
                'class': 'logging.FileHandler',
                'filename': log_file,
                'level': log_level,
                'formatter': 'structured' if structured_logging else 'standard',
                'mode': 'a'
            }
        
        root_handlers = ['console']
        if log_file:
            root_handlers.append('file')
        
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': formatters,
            'handlers': handlers,
            'root': {
                'level': log_level,
                'handlers': root_handlers
            },
            'loggers': {
                'rl4sys': {
                    'level': log_level,
                    'handlers': root_handlers,
                    'propagate': False
                },
                'rl4sys.server': {
                    'level': log_level,
                    'handlers': root_handlers,
                    'propagate': False
                },
                'rl4sys.client': {
                    'level': log_level,
                    'handlers': root_handlers,
                    'propagate': False
                },
                'rl4sys.algorithms': {
                    'level': log_level,
                    'handlers': root_handlers,
                    'propagate': False
                }
            }
        }
        
        return config
    
    @staticmethod
    def setup_logging(
        config_dict: Optional[Dict[str, Any]] = None,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        debug: bool = False
    ) -> None:
        """
        Set up logging for the entire RL4Sys framework.
        
        Args:
            config_dict: Custom logging configuration dictionary
            log_level: Default log level if config_dict is None
            log_file: Log file path if config_dict is None
            debug: Enable debug mode (sets log level to DEBUG)
        """
        
        if debug:
            log_level = "DEBUG"
        
        if config_dict is None:
            config_dict = RL4SysLogConfig.get_default_config(
                log_level=log_level,
                log_file=log_file,
                structured_logging=False
            )
        
        logging.config.dictConfig(config_dict)
    
    @staticmethod
    def get_environment_config() -> Dict[str, str]:
        """
        Get logging configuration from environment variables.
        
        Supported environment variables:
        - RL4SYS_LOG_LEVEL: Logging level (default: INFO)
        - RL4SYS_LOG_FILE: Log file path (default: None)
        - RL4SYS_DEBUG: Enable debug mode (default: False)
        
        Returns:
            Dictionary with environment-based configuration
        """
        return {
            'log_level': os.getenv('RL4SYS_LOG_LEVEL', 'INFO').upper(),
            'log_file': os.getenv('RL4SYS_LOG_FILE'),
            'debug': os.getenv('RL4SYS_DEBUG', 'false').lower() in ('true', '1', 'yes')
        }

def setup_rl4sys_logging(debug: bool = False, log_file: Optional[str] = None) -> None:
    """
    Convenience function to set up RL4Sys logging with environment variable support.
    
    Args:
        debug: Enable debug mode
        log_file: Optional log file path
    """
    env_config = RL4SysLogConfig.get_environment_config()
    
    # Command line arguments override environment variables
    actual_debug = debug or env_config['debug']
    actual_log_file = log_file or env_config['log_file']
    actual_log_level = env_config['log_level'] if not actual_debug else 'DEBUG'
    
    RL4SysLogConfig.setup_logging(
        log_level=actual_log_level,
        log_file=actual_log_file,
        debug=actual_debug
    )

# Performance optimization: logging guards
def should_log_debug() -> bool:
    """Check if debug logging is enabled to avoid expensive operations."""
    return logging.getLogger('rl4sys').isEnabledFor(logging.DEBUG)

def should_log_info() -> bool:
    """Check if info logging is enabled to avoid expensive operations."""
    return logging.getLogger('rl4sys').isEnabledFor(logging.INFO)