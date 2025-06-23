import os
import sys
import time
import threading
import io
import json
import datetime
from pathlib import Path
from queue import Empty, Queue
import zlib

import torch
from numpy import ndarray
import numpy as np
import random
import grpc

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rl4sys.algorithms.DQN.kernel import DeepQNetwork
from rl4sys.algorithms.PPO.kernel import RLActorCritic
from rl4sys.common.action import RL4SysAction
from rl4sys.common.trajectory import RL4SysTrajectory
from rl4sys.utils.util import serialize_action, StructuredLogger
from rl4sys.proto import (
    GetModelRequest,
    SendTrajectoriesRequest,
    Trajectory,
    RLServiceStub,
    InitRequest,
    ParameterValue
)
from rl4sys.client.config_loader import AgentConfigLoader

MODEL_CLASSES = {
    'PPO': RLActorCritic,
    'DQN': DeepQNetwork,
}

class RL4SysAgent:
    def __init__(self, conf_path: str, debug: bool = False):
        """
        Initialize the RL4SysAgent with a configuration file.
        
        Args:
            conf_path (str): Path to the JSON configuration file
            debug (bool): Whether to enable debug logging
        """
        self.agent_config_loader = AgentConfigLoader(conf_path)
        self.client_id = self.agent_config_loader.get_client_id()
        self.algorithm_name = self.agent_config_loader.get_algorithm_name()
        self.algorithm_parameters = self.agent_config_loader.get_algorithm_parameters()
        self.algorithm_type = self.agent_config_loader.get_algorithm_type()

        self.debug = debug
        self.logger = StructuredLogger("RL4SysAgent", debug)
        
        # get server address
        self.server_address = self.agent_config_loader.get_train_server_address()

        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = RLServiceStub(self.channel)

        # Initialize model and version tracking
        self._model = None
        self.local_version = 0
        self._lock = threading.Lock()
        self._trajectory_lock = threading.Lock()

        # initialize the trajectory buffer, it should store multiple trajectories
        self._trajectory_buffer = []
        self._trajectory_buffer_size = 0
        self._trajectory_send_threshold = self.agent_config_loader.get_send_frequency()

        # Initialize thread for sending trajectories
        self._send_thread = None
        self._stop_event = threading.Event()
        self._send_queue = Queue()

        # init server algorithm
        self._init_server_algorithm(self.client_id, 
                                    self.algorithm_name, 
                                    self.algorithm_parameters)
        

        # Get initial model from server
        self.logger.info("Getting initial model from server")
        with self._lock:
            self._model, self.local_version = self._get_model_unsafe(expected_version=-1)
        assert self._model is not None, "Model should be loaded from server at this point"

        self.logger.info(
            "Successfully loaded model",
            model_name=self._model.get_model_name(),
            version=self.local_version
        )

        # Create debug directory if it doesn't exist
        self.debug_dir = Path('debug')
        self.debug_dir.mkdir(exist_ok=True)

        # Start the sending thread
        self._start_send_thread()

    def _init_server_algorithm(self, 
                               client_id: str, 
                               algorithm_name: str, 
                               algorithm_parameters: dict):
        """Initialize the server-side algorithm."""

        self.logger.info("Initializing server-side algorithm", 
                         client_id=client_id, 
                         algorithm_name=algorithm_name)
        
        # Convert Python types to ParameterValue
        param_values = {}
        for key, value in algorithm_parameters.items():
            param_value = ParameterValue()
            if isinstance(value, int):
                param_value.int_value = value
            elif isinstance(value, float):
                param_value.float_value = value
            elif isinstance(value, str):
                param_value.string_value = value
            elif isinstance(value, bool):
                param_value.bool_value = value
            # null values are handled by not setting any field
            
            param_values[key] = param_value
        
        request = InitRequest(
            client_id=client_id, 
            algorithm_name=algorithm_name, 
            algorithm_parameters=param_values
        )

        response = self.stub.InitAlgorithm(request)
        if not response.is_success:
            raise RuntimeError(f"Failed to initialize algorithm: {response.message}")

    def _start_send_thread(self):
        """Start the thread for sending trajectories."""
        self._send_thread = threading.Thread(target=self._send_thread_worker)
        self._send_thread.daemon = True
        self._send_thread.start()
        self.logger.debug("Started trajectory sending thread")

    def _send_thread_worker(self):
        """Worker function for the sending thread."""
        while not self._stop_event.is_set():
            try:
                # Block with timeout until there are trajectories to send or shutdown is requested
                try:
                    trajectories = self._send_queue.get(timeout=1.0)  # 1 second timeout
                    if trajectories:
                        self._send_trajectories(trajectories)
                        # Clear trajectories after sending to free memory
                        for traj in trajectories:
                            traj.clear()
                        trajectories.clear()
                    self._send_queue.task_done()
                except Empty:
                    # Timeout occurred, check stop_event and continue
                    continue
            except Exception as e:
                self.logger.error(
                    "Error in send thread",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                # If there's an error, wait a bit before retrying
                time.sleep(1)

    def _check_and_send_trajectories(self):
        """Check if we need to send trajectories and queue them if needed."""
        with self._trajectory_lock:
            # Count only completed trajectories
            completed_trajectories = [t for t in self._trajectory_buffer if t.completed]
            completed_count = len(completed_trajectories)
            
            # Only send if we have enough completed trajectories
            if completed_count >= self._trajectory_send_threshold:
                self.logger.info(
                    "Sending completed trajectories",
                    count=completed_count,
                    threshold=self._trajectory_send_threshold
                )
                # Remove completed trajectories from buffer
                self._trajectory_buffer = [t for t in self._trajectory_buffer if not t.completed]
                self._trajectory_buffer_size = len(self._trajectory_buffer)
                # Queue completed trajectories for sending
                self._send_queue.put(completed_trajectories)

    def request_for_action(self, 
                          traj: RL4SysTrajectory = None, 
                          obs: torch.Tensor = None, 
                          *args, **kwargs):
        """
        Produce an action from the current model. Stores the action in our local trajectory buffer.
        """
        # Get current version safely before creating trajectory
        with self._lock:
            current_version = self.local_version
            
        if traj is None or traj.is_completed():
            traj = RL4SysTrajectory(version=current_version)
            with self._trajectory_lock:
                self._trajectory_buffer.append(traj)
                self._trajectory_buffer_size += 1
                self.logger.debug(
                    "Created new trajectory",
                    version=traj.version,
                    buffer_size=self._trajectory_buffer_size
                )

        if obs is None:
            raise ValueError("Observation is required")

        with self._lock:
            if self._model is None:
                raise RuntimeError("No model available yet!")

            action_nd, data_dict = self._model.step(obs)
            self.logger.debug(
                "Generated action",
                action_shape=action_nd.shape,
                data_keys=list(data_dict.keys())
            )

        action = RL4SysAction(obs, action_nd, reward=-1, done=False, data=data_dict, version=traj.version)
        
        return traj, action

    def add_to_trajectory(self, traj: RL4SysTrajectory, action: RL4SysAction):
        """
        Add an action to the current trajectory
        """
        traj.add_action(action)
        self.logger.debug(
            "Added action to trajectory",
            trajectory_version=traj.version,
            action_count=len(traj.actions)
        )

    def mark_end_of_trajectory(self, traj: RL4SysTrajectory, action: RL4SysAction):
        """
        Mark the end of the current trajectory
        """
        action.done = True
        traj.mark_completed()
        self.logger.debug(
            "Marked trajectory as completed",
            version=traj.version,
            action_count=len(traj.actions)
        )
        # Check if we need to send trajectories after marking completion
        self._check_and_send_trajectories()

    def update_action_reward(self, action: RL4SysAction, reward: float):
        """
        Update the reward of the current action
        """
        action.reward = reward
        self.logger.debug(
            "Updated action reward",
            reward=reward,
            version=action.version
        )

    def _send_trajectories(self, trajectories) -> int:
        """
        Send given trajectories to the server. 
        This function is called by the sending thread, so it will not block the main thread.
        Returns:
            int: 0 if model was not updated, otherwise the version of the model
        """
        if not trajectories:
            self.logger.debug("No trajectories to send")
            return 0

        try:
            # Get algorithm type from config
            is_onpolicy = self.algorithm_type == 'onpolicy'
            
            # Filter trajectories based on algorithm type and validity
            # Need to acquire lock to safely read local_version
            with self._lock:
                current_version = self.local_version
            
            if is_onpolicy:
                # For on-policy algorithms, only send valid trajectories with matching version
                filtered_trajectories = [t for t in trajectories if t.version == current_version and t.is_valid()]
                ignored_count = len(trajectories) - len(filtered_trajectories)
                if ignored_count > 0:
                    self.logger.debug(
                        "Ignored trajectories with version mismatch",
                        count=ignored_count,
                        current_version=current_version
                    )
                trajectories = filtered_trajectories
            else:
                # For off-policy algorithms, only send valid trajectories
                filtered_trajectories = [t for t in trajectories if t.is_valid()]
                ignored_count = len(trajectories) - len(filtered_trajectories)
                if ignored_count > 0:
                    self.logger.debug(
                        "Ignored invalid trajectories",
                        count=ignored_count
                    )
                trajectories = filtered_trajectories

            if not trajectories:
                self.logger.debug("No valid trajectories to send after filtering")
                return 0

            # Convert trajectories to protobuf format
            trajectories_proto = []
            for traj in trajectories:
                # Convert each action in the trajectory to protobuf format
                actions_proto = []
                for action in traj.actions:
                    action_proto = serialize_action(action)
                    actions_proto.append(action_proto)
                
                # Create trajectory protobuf
                traj_proto = Trajectory(
                    actions=actions_proto,
                    version=traj.version
                )
                trajectories_proto.append(traj_proto)

            # Create request
            request = SendTrajectoriesRequest(
                client_id=self.client_id,
                trajectories=trajectories_proto
            )

            # Send to server
            response = self.stub.SendTrajectories(request)
            
            if response.model_updated:
                # Need to update model and version together under lock
                with self._lock:
                    old_version = self.local_version
                self.logger.info(
                    "Model updated",
                        old_version=old_version,
                    new_version=response.new_version
                )
                # Get new model (this also updates self._model internally)
                self._model, _ = self._get_model_unsafe(response.new_version)
                # update local version
                self.local_version = response.new_version
                return response.new_version
            else:
                self.logger.debug("No model update needed")
                return 0

        except grpc.RpcError as e:
            self.logger.error(
                "Failed to send trajectories",
                error=e.details(),
                error_type=type(e).__name__
            )
            return 0

    def _get_model(self, expected_version: int):
        """Get the model from the server with proper thread synchronization."""
        with self._lock:
            return self._get_model_unsafe(expected_version)
    
    def _get_model_unsafe(self, expected_version: int):
        """Get the model from the server. Must be called with _lock held."""
        try:
            # Request the latest model
            request = GetModelRequest(client_id=self.client_id, client_version=self.local_version, expected_version=expected_version)
            response = self.stub.GetModel(request)
            
            # If we got an empty response, it means we already have the latest version
            if len(response.model_state) == 0:
                self.logger.debug(
                    "Already have latest model version",
                    client_version=self.local_version,
                    expected_version=expected_version,
                )
                return self._model, response.version
                
            # Verify version if needed
            if expected_version != -1 and response.version != expected_version:
                self.logger.error(
                    "Version mismatch",
                    expected=expected_version,
                    received=response.version
                )
                return None, -1
                
            model_class = MODEL_CLASSES.get(self.algorithm_name)
            if model_class is None:
                raise ValueError(f"Unsupported algorithm: {self.algorithm_name}")
            
            # Create new model instance if needed
            if self._model is None:
                model_input_size = self.algorithm_parameters['input_size']
                model_act_dim = self.algorithm_parameters['act_dim']
                self._model = model_class(model_input_size, model_act_dim)

            # Decompress the model state
            try:
                decompressed = zlib.decompress(response.model_state)
                buffer = io.BytesIO(decompressed)
                loaded_obj = torch.load(buffer, weights_only=False)
            except zlib.error:
                # If decompression fails, try loading directly (backward compatibility)
                buffer = io.BytesIO(response.model_state)
                loaded_obj = torch.load(buffer, weights_only=False)
            
            if response.is_diff:
                # For diff updates, merge with current state (all under lock)
                current_state = self._model.state_dict()
                for key, value in loaded_obj.items():
                    current_state[key] = value
                self._model.load_state_dict(current_state)
                self.logger.debug(
                    "Applied model diff",
                    changed_params=len(loaded_obj),
                    version=response.version
                )
            else:
                # For full model updates, replace the entire state dict 
                self._model.load_state_dict(loaded_obj)
                self.logger.debug(
                    "Loaded complete model state",
                    version=response.version
                )
            
            self.logger.info(
                "Successfully loaded model",
                algorithm=self.algorithm_name,
                version=response.version
            )

            # TODO debug only
            # print(f"model weights: {self._model.state_dict()}")

            return self._model, response.version
            
        except grpc.RpcError as e:
            self.logger.error(
                "Failed to get model from server",
                error=e.details(),
                error_type=type(e).__name__
            )
            return None, -1

    def close(self):
        """Cleanly close the gRPC channel and stop the sending thread."""
        self._stop_event.set()
        if self._send_thread:
            self._send_thread.join()
        self.channel.close()
        self.logger.info("Closed gRPC channel and stopped sending thread")
