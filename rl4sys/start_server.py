#!/usr/bin/env python3
"""
RL4Sys Server Starter Script

This script starts the RL4Sys gRPC server for model serving and trajectory collection.
"""

import os
import sys
import grpc
from concurrent import futures
import logging
import argparse

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "proto"))

from rl4sys.server.server import MyRLServiceServicer
from rl4sys.proto.rl4sys_pb2_grpc import add_RLServiceServicer_to_server
from rl4sys.utils.logging_config import setup_rl4sys_logging
from rl4sys.utils.util import StructuredLogger

def start_server(port=50051, max_workers=10, debug=False):
    """
    Start the RL4Sys gRPC server.
    
    Args:
        port (int): Port number to bind the server to
        max_workers (int): Maximum number of worker threads for handling RPCs
        debug (bool): Enable debug logging
    """
    # Configure centralized logging with structured format to show metrics
    from rl4sys.utils.logging_config import RL4SysLogConfig
    config = RL4SysLogConfig.get_default_config(
        log_level="DEBUG" if debug else "INFO",
        structured_logging=True  # Enable structured logging to show metric numbers
    )
    RL4SysLogConfig.setup_logging(config_dict=config)
    logger = StructuredLogger('RL4SysServer', debug=debug)
    
    if debug:
        logger.debug("Debug mode enabled - verbose logging is active")
    
    try:
        # Create gRPC server with 32MB message size limit
        server_options = [
            ('grpc.max_send_message_length', 200 * 1024 * 1024),  # 200MB
            ('grpc.max_receive_message_length', 200 * 1024 * 1024),  # 200MB
        ]
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=server_options
        )
        
        # Add RLServiceServicer to the server with debug flag
        servicer = MyRLServiceServicer(debug=debug)
        add_RLServiceServicer_to_server(servicer, server)
        
        # Bind server to port
        server.add_insecure_port(f'[::]:{port}')
        
        # Start server
        server.start()
        logger.info(f"RL4Sys server started on port {port}", 
                   port=port,
                   max_send_message_length="200MB",
                   max_receive_message_length="200MB",
                   compression="Gzip")
        
        # Keep server running
        server.wait_for_termination()
        
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)
        logger.info("Server shut down successfully")
    except Exception as e:
        logger.error(f"Error starting server: {e}", error=str(e))
        raise

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='RL4Sys Training Server')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--port', type=int, default=50051, help='Server port')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker threads')
    args = parser.parse_args()
    
    # Start server with parsed arguments
    start_server(port=args.port, max_workers=args.workers, debug=args.debug) 