import io
import torch
import grpc
from concurrent import futures
import time
import threading
from queue import Queue, Empty
import logging
import argparse
import json
import hashlib
import zlib
from typing import Dict, Tuple, Optional, List, Any
import os
import numpy as np
import random
from rl4sys.proto import (
    ModelResponse,
    SendTrajectoriesResponse,
    Trajectory,
    Action,
    InitRequest,
    InitResponse
)
from rl4sys.proto.rl4sys_pb2_grpc import RLServiceServicer
from rl4sys.algorithms.PPO.PPO import PPO
from rl4sys.algorithms.PPO_Continuous.PPO_cont import PPOCont
from rl4sys.algorithms.DQN.DQN import DQN
from rl4sys.utils.util import deserialize_action, StructuredLogger
from rl4sys.server.model_diff_manager import ModelDiffManager
from rl4sys.utils.system_monitor import SystemMonitor, log_memory_usage
from rl4sys.utils.packet_logger import PacketLogger

import csv

# Algorithm class mapping
ALGORITHM_CLASSES = {
    'PPO': PPO,
    'PPO_Continuous': PPOCont,
    'DQN': DQN,
}

class ClientAlgorithmManager:
    """Manages algorithm instances for different clients."""
    def __init__(self):
        self.logger = StructuredLogger("ClientAlgorithmManager", debug=False)

        # Dictionary to store algorithm instances per client
        self.client_algorithms = {}
        # Dictionary to store model diff managers per client
        self.client_diff_managers = {}
        # Dictionary to store training queues per client
        self.client_training_queues = {}
        # Dictionary to store training threads per client
        self.client_training_threads = {}
        # Dictionary to store stop events per client
        self.client_stop_events = {}
        
        # Thread synchronization
        self._algorithm_lock = threading.Lock()
        
        self.logger.info(
            "Initialized ClientAlgorithmManager"
        )
    
    def is_client_exists(self, client_id: str) -> bool:
        """Check if a client exists."""
        with self._algorithm_lock:
            return client_id in self.client_algorithms
    
    def get_algorithm(self, client_id: str) -> Tuple[Any, ModelDiffManager]:
        """Get or create an algorithm instance for a specific client."""
        with self._algorithm_lock:
            if client_id not in self.client_algorithms:
                self.create_algorithm(client_id)
            return self.client_algorithms[client_id], self.client_diff_managers[client_id]
    
    def create_algorithm(self, 
                         client_id: str, 
                         algorithm_name: str, 
                         modelparameters: dict):
        """Create a new algorithm instance for a client."""
        self.logger.info(
            "Creating new algorithm instance for client",
            client_id=client_id,
            algorithm_name=algorithm_name
        )
        
        # Get the algorithm class
        algorithm_class = ALGORITHM_CLASSES.get(algorithm_name)
        if algorithm_class is None:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")
        
        # Initialize algorithm with common parameters and algorithm-specific parameters
        algorithm = algorithm_class(
            version=0,
            **modelparameters
        )
        
        # Create and initialize model diff manager for this client
        diff_manager = ModelDiffManager()
        diff_manager.add_model_version(0, algorithm.get_current_model()[0])
        
        # Create training queue and stop event for this client
        training_queue = Queue()
        stop_event = threading.Event()
        
        # Create and start training thread for this client
        training_thread = threading.Thread(
            target=self._client_training_loop,
            args=(client_id, training_queue, stop_event),
            name=f"TrainingThread-{client_id}",
            daemon=True
        )
        training_thread.start()
        
        # Store in dictionaries
        self.client_algorithms[client_id] = algorithm
        self.client_diff_managers[client_id] = diff_manager
        self.client_training_queues[client_id] = training_queue
        self.client_training_threads[client_id] = training_thread
        self.client_stop_events[client_id] = stop_event
        
        self.logger.info(
            "Algorithm initialized for client",
            client_id=client_id,
            algorithm_name=algorithm_name,
            version=0
        )
    
    def _client_training_loop(self, client_id: str, training_queue: Queue, stop_event: threading.Event):
        """Training loop for a specific client's algorithm."""
        thread_name = threading.current_thread().name
        self.logger.info(f"Training thread {thread_name} started for client {client_id}", thread_name=thread_name, client_id=client_id)
        
        while not stop_event.is_set():
            try:
                # Get trajectory from queue with timeout
                traj_data = training_queue.get(timeout=1.0)
                if traj_data is None:
                    continue
                
                traj, version = traj_data

                """
                print(f"traj: {traj[-1].obs.tolist()}")
                print(f"traj: {traj[-1].mask.tolist()}")
                print(f"traj: {traj[-1].act}")
                print(f"traj: {traj[-1].rew}")
                print(f"traj: {traj[-1].done}")
                """

                
                # Get the algorithm instance for this client
                algorithm, diff_manager = self.get_algorithm(client_id)
                
                self.logger.debug(
                    "Processing trajectory",
                    thread_name=thread_name,
                    client_id=client_id,
                    trajectory_version=version,
                    current_version=algorithm.version,
                    action_count=len(traj)
                )
                
                # Process trajectory and update model
                if algorithm.type == 'onpolicy':
                    if version == algorithm.version:
                        algorithm.receive_trajectory(traj, version)
                        # Add new model version to history after update
                        model, new_version = algorithm.get_current_model()
                        diff_manager.add_model_version(new_version, model)
                    else:
                        self.logger.debug(
                            "Skipping on-policy trajectory due to version mismatch",
                            thread_name=thread_name,
                            client_id=client_id,
                            trajectory_version=version,
                            current_version=algorithm.version
                        )
                else:
                    algorithm.receive_trajectory(traj)
                    # Add new model version to history after update
                    model, new_version = algorithm.get_current_model()
                    diff_manager.add_model_version(new_version, model)
                
                training_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                self.logger.error(
                    "Error processing trajectory",
                    thread_name=thread_name,
                    client_id=client_id,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                continue
    
    def get_client_ids(self) -> List[str]:
        """Get list of all client IDs."""
        with self._algorithm_lock:
            return list(self.client_algorithms.keys())
    
    def remove_client(self, client_id: str):
        """Remove a client's algorithm instance and cleanup resources."""
        with self._algorithm_lock:
            if client_id in self.client_algorithms:
                # Signal stop event and wait for thread to finish
                self.client_stop_events[client_id].set()
                self.client_training_threads[client_id].join()
                
                # Cleanup resources
                del self.client_algorithms[client_id]
                del self.client_diff_managers[client_id]
                del self.client_training_queues[client_id]
                del self.client_training_threads[client_id]
                del self.client_stop_events[client_id]
                
                self.logger.info("Removed client", client_id=client_id)
    
    def dispatch_trajectory(self, client_id: str, trajectory: List[Any], version: int):
        """Dispatch a trajectory to the appropriate client's training queue."""
        if client_id in self.client_training_queues:
            self.client_training_queues[client_id].put((trajectory, version))
            return True
        return False

# Define gRPC service
class MyRLServiceServicer(RLServiceServicer):
    def __init__(self, debug=False, num_workers=4):
        self.debug = debug
        self.logger = StructuredLogger("RL4SysServer", debug)
        
        # Initialize system monitoring
        self.system_monitor = SystemMonitor(
            "RL4SysServer", 
            debug=debug,
            save_to_file=True,
            project_name="rl4sys_server"
        )
        self.system_monitor.start_monitoring()
        
        # ------------------------------------------------------------------
        # Packet logger – records the number of trajectories ("packets")
        # received by the server.  Logging interval set to 5 seconds.
        # ------------------------------------------------------------------
        self.packet_logger: PacketLogger = PacketLogger(
            project_name="rl4sys_server_21_client", # TODO change the name to the number of clients
            log_interval=0.5,
            debug=debug
        )
        
        self.logger.info(
            "Initializing multi-client server",
            num_workers=num_workers
        )
        
        # Log initial memory usage
        log_memory_usage(self.logger, "at server startup")
        
        # Initialize client algorithm manager
        self.client_manager = ClientAlgorithmManager()
        
        # Initialize dispatcher control event
        self._stop_event = threading.Event()
        
        # Initialize trajectory processing queue and attach to packet logger for monitoring
        self.trajectory_queue = Queue()
        # PacketLogger monitors queue size and throughput
        self.packet_logger.attach_queue(self.trajectory_queue)
        
        # Create dispatcher thread
        self._dispatcher_thread = threading.Thread(
            target=self._dispatch_trajectories,
            name="TrajectoryDispatcher",
            daemon=True
        )
        self._dispatcher_thread.start()
        
        self.logger.info(
            "Multi-client server ready",
            supported_algorithms=list(ALGORITHM_CLASSES.keys())
        )
        
    def _dispatch_trajectories(self):
        """Central dispatcher thread that distributes trajectories to client-specific training threads."""
        thread_name = threading.current_thread().name
        self.logger.info(f"Trajectory dispatcher thread {thread_name} started", thread_name=thread_name)
        
        while not self._stop_event.is_set():
            try:
                # Get trajectory from queue with timeout
                traj_data = self.trajectory_queue.get(timeout=1.0)
                if traj_data is None:
                    continue
                
                traj, version, client_id = traj_data
                
                # Dispatch trajectory to client-specific training thread
                if not self.client_manager.dispatch_trajectory(client_id, traj, version):
                    self.logger.warning(
                        "Failed to dispatch trajectory - client not found",
                        client_id=client_id
                    )
                
                self.trajectory_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                self.logger.error(
                    "Error dispatching trajectory",
                    thread_name=thread_name,
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                continue

    def InitAlgorithm(self, request, context):
        """Initialize an algorithm for a client with RPC compression."""
        # Enable compression for this response
        context.set_compression(grpc.Compression.Gzip)
        
        client_id = request.client_id
        algorithm_name = request.algorithm_name
        algorithm_parameters = {}
        
        # Convert ParameterValue to Python types
        for key, param_value in request.algorithm_parameters.items():
            if param_value.HasField('int_value'):
                value = param_value.int_value
            elif param_value.HasField('float_value'):
                value = param_value.float_value
            elif param_value.HasField('string_value'):
                value = param_value.string_value
            elif param_value.HasField('bool_value'):
                value = param_value.bool_value
            else:
                value = None  # null value
            
            algorithm_parameters[key] = value

        # set numpy, random torch seed
        np.random.seed(algorithm_parameters['seed'])
        random.seed(algorithm_parameters['seed'])
        torch.manual_seed(algorithm_parameters['seed'])
        
        # Check if client already exists
        if self.client_manager.is_client_exists(client_id):
            self.logger.info(
                "Algorithm exists, reuse it",
                client_id=client_id,
                algorithm_name=algorithm_name
            )
            return InitResponse(is_success=True, message="Algorithm exists, reuse it")
        
        # Create new algorithm for client
        self.client_manager.create_algorithm(client_id, 
                                           algorithm_name, 
                                           algorithm_parameters)
        
        self.logger.info(
            "Algorithm initialized for client",
            client_id=client_id,
            algorithm_name=algorithm_name,
            parameters=algorithm_parameters
        )
        
        return InitResponse(is_success=True, message="Algorithm initialized successfully")

    def GetModel(self, request, context):
        """Get model updates with RPC compression for large model states."""
        # Enable compression for this response (especially important for model data)
        context.set_compression(grpc.Compression.Gzip)
        
        client_id = request.client_id
        client_version = request.client_version
        expected_version = request.expected_version
        
        # Get or create algorithm for this client
        algorithm, diff_manager = self.client_manager.get_algorithm(client_id)
        
        model, current_version = algorithm.get_current_model()

        #print(f"model before send: {model.state_dict()}")
        
        # Add current model to history if not already there
        diff_manager.add_model_version(current_version, model)
        
        # If client is requesting from scratch (version -1), return complete model
        if expected_version == -1:
            self.logger.debug(
                "Client requesting complete model",
                client_id=client_id,
                version=current_version
            )
            compressed_state = diff_manager._compress_state_dict(model.state_dict())
            return ModelResponse(version=current_version, is_diff=False, model_state=compressed_state)
        
        # Get model diff or full model
        model_data = diff_manager.get_model_diff(client_version, expected_version, current_version)
        
        if model_data is None:
            self.logger.debug(
                "Client already has the expected version",
                client_id=client_id,
                version=expected_version
            )
            return ModelResponse(version=expected_version, is_diff=False, model_state=b"")
        
        model_state, version = model_data

        """
        diff_bytes  = len(model_state)
        full_state  = diff_manager._compress_state_dict(model.state_dict())
        full_bytes  = len(full_state)
        saved_bytes = full_bytes - diff_bytes
        save_ratio  = saved_bytes / full_bytes if full_bytes else 0.0

        # append one line to a CSV file
        with open("model_size_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), client_id, current_version,
                diff_bytes, full_bytes, saved_bytes, save_ratio
            ])
        """        


        self.logger.info(
            "Sending model update to client",
            client_id=client_id,
            client_version=client_version,
            expected_version=expected_version,
            model_size=len(model_state)
        )
        # print(f"model_state: {model.state_dict()}")
        return ModelResponse(version=version, is_diff=True, model_state=model_state)

    def SendTrajectories(self, request, context):
        """Send trajectories with RPC compression for large trajectory data."""
        # Enable compression for this response
        context.set_compression(grpc.Compression.Gzip)
        
        client_id = request.client_id
        
        # Get or create algorithm for this client
        algorithm, _ = self.client_manager.get_algorithm(client_id)
        
        traj_count = len(request.trajectories)

        # Increment packet logger with the number of trajectories received
        self.packet_logger.increment(traj_count)
        
        self.logger.debug(
            "SendTrajectories request",
            client_id=client_id,
            trajectory_count=traj_count
        )
        
        for traj_idx, traj_proto in enumerate(request.trajectories):
            # Convert protobuf trajectory to RL4SysTrajectory
            traj = []
            for action_proto in traj_proto.actions:
                action = deserialize_action(action_proto)
                traj.append(action)
            
            self.logger.debug(
                "Processing trajectory",
                client_id=client_id,
                trajectory_index=traj_idx + 1,
                total_trajectories=traj_count,
                version=traj_proto.version,
                action_count=len(traj_proto.actions)
            )
            
            # Add trajectory to processing queue
            self.trajectory_queue.put((traj, traj_proto.version, client_id))
            # Record queue size right after the put for precise backlog tracking
            self.packet_logger.update_queue_size(self.trajectory_queue.qsize())

        # Check if we need to update the model
        max_client_version = max(traj.version for traj in request.trajectories)
        updated = max_client_version < algorithm.version
        current_version = algorithm.version
        
        self.logger.debug(
            "Sending response to client",
            client_id=client_id,
            model_updated=updated,
            new_version=current_version
        )
        
        return SendTrajectoriesResponse(
            model_updated=updated,
            new_version=current_version
        )

    def __del__(self):
        """Cleanup when the service is destroyed."""
        self.logger.info("Shutting down server")
        self._stop_event.set()
        
        # Wait for all worker threads to complete
        self._dispatcher_thread.join()
        
        # Stop packet logger to flush remaining logs
        if hasattr(self, "packet_logger"):
            self.packet_logger.stop()
            
        self.logger.info("Server shutdown complete")

# Start gRPC server
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL4Sys Training Server')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--port', type=int, default=50051, help='Server port')
    args = parser.parse_args()
    
    # Configure server options with compression and 32MB message size
    options = [
        ('grpc.default_compression_algorithm', grpc.Compression.Gzip),
        ('grpc.compression_level', 6),  # High compression level (0-9)
        ('grpc.so_reuseport', 1),
        ('grpc.max_send_message_length', 32 * 1024 * 1024),  # 32MB
        ('grpc.max_receive_message_length', 32 * 1024 * 1024),  # 32MB
    ]
    
    # Create server with compression options
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=options
    )
    servicer = MyRLServiceServicer(debug=args.debug, num_workers=8)
    
    # Missing variable pb2_grpc - fix import
    from rl4sys.proto.rl4sys_pb2_grpc import add_RLServiceServicer_to_server
    add_RLServiceServicer_to_server(servicer, server)
    
    # Start server
    server_address = f'[::]:{args.port}'
    server.add_insecure_port(server_address)
    server.start()
    servicer.logger.info(
        "Server started with gRPC compression enabled",
        port=args.port,
        debug_mode=args.debug,
        compression_algorithm="Gzip",
        compression_level=6
    )
    
    # Block until the server is terminated
    server.wait_for_termination()