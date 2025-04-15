import sys
import os
import time
import threading
from pathlib import Path

import torch
from numpy import ndarray

from protocol.trajectory import RL4SysTrajectory
from protocol.action import RL4SysAction

from client.client import RL4SysClient

import random
import numpy as np

class RL4SysAgent:
    """
    Trajectory-collecting agent that:
      - Uses models to generate actions for the environment
      - Collects trajectories and sends them to a client
      - Model updates are pushed directly from the client's streaming thread
      - Trajectories are sent in batches for efficiency
      - Can randomly sample a subset of trajectories to send when buffer is large
    """

    def __init__(self, 
                 algorithm_name: str,
                 model: torch.nn.Module = None, 
                 input_size: int = 0,
                 act_dim: int = 0,
                 act_limit: float = 1.0,
                 min_trajectories_to_send: int = 5,
                 max_trajectories_to_send: int = 10,
                 verbose: bool = True):
        """
        Args:
            algorithm_name: Name of the RL algorithm to use
            model: Optional initial model to use
            input_size: Size of the observation space
            act_dim: Size of the action space
            act_limit: Limit for continuous action values
            min_trajectories_to_send: Minimum number of trajectories to accumulate before sending
            max_trajectories_to_send: Maximum number of trajectories to send in one batch (samples if more are available)
            verbose: Whether to print status messages
        """
        self.algorithm_name = algorithm_name
        self._model = model
        self._lock = threading.Lock()
        self.verbose = verbose

        # Initialize trajectory buffer
        self._current_traj = RL4SysTrajectory()
        
        # Initialize client for server communication, passing self reference
        # for direct model updates
        self.client = RL4SysClient(
            algorithm_name=algorithm_name,
            input_size=input_size,
            act_dim=act_dim,
            act_limit=act_limit,
            agent_reference=self,  # Pass self reference for direct model updates
            min_trajectories_to_send=min_trajectories_to_send,  # Configure minimum batch size
            max_trajectories_to_send=max_trajectories_to_send,  # Configure maximum batch size
            verbose=verbose  # Pass verbose flag to client
        )
        
        # Set running flag for cleanup
        self._running = True

    def _validate_model(self, model: torch.nn.Module) -> None:
        """Check that the model has proper interface for generating actions."""
        assert hasattr(model, 'step'), "Model must have a .step(...) method."
        # During actual usage, the model validity will be verified by usage patterns

    def request_for_action(self, obs: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> RL4SysAction:
        """
        Produce an action from the current model. Stores the action in our local trajectory buffer.
        """
        with self._lock:
            if self._model is None:
                # If no model available, initialize with random action and empty data
                if self.verbose:
                    print("[RL4SysAgent] Warning: No model available, using random action.")
                if self.algorithm_name == "DQN":
                    action_nd = np.random.randint(0, mask.shape[1] if mask is not None else 1)
                else:
                    action_nd = np.random.uniform(-1, 1, size=(1,))
                data_dict = {}
            else:
                # Use the model to generate an action
                if self.algorithm_name == "DQN":
                    action_nd, data_dict = self._model.step(obs, mask=mask)
                elif self.algorithm_name == "PPO":
                    action_nd, logp_a, _, value = self._model.get_action_and_value(obs, mask=mask)
                    data_dict = {}
                    data_dict['logp_a'] = logp_a
                    data_dict['v'] = value
                    action_nd = action_nd.numpy()
                elif self.algorithm_name == "DDPG":
                    action_nd, data_dict = self._model.step(obs, mask=mask)
                elif self.algorithm_name == "RPO":
                    action_nd, data_dict = self._model.get_action_and_value(obs, mask=mask)

        # Create an action object and add it to the current trajectory
        r4sa = RL4SysAction(obs, action_nd, mask=mask, reward=-1, data=data_dict, done=False)
        self._current_traj.add_action(r4sa)
        return r4sa

    def send_actions(self) -> None:
        """
        Mark the end of the current trajectory and queue it for sending to the server.
        Non-blocking operation. The client will collect trajectories and send them
        in batches when appropriate.
        """
        # Only add to queue if the trajectory contains actions
        if len(self._current_traj.actions) > 0:
            # Add the current trajectory to the client's trajectory buffer
            self.client.add_trajectory(self._current_traj)
            if self.verbose:
                print(f"[RL4SysAgent] Trajectory added to client's buffer (length: {len(self._current_traj.actions)} actions)")
            
            # Start a new trajectory
            self._current_traj = RL4SysTrajectory()
        else:
            if self.verbose:
                print("[RL4SysAgent] Skipping empty trajectory")

    def close(self):
        """Cleanly close connections and threads."""
        self._running = False
        
        # Close client connection
        self.client.close()
        if self.verbose:
            print("[RL4SysAgent] Agent closed successfully.")
