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
      - Periodically checks for model updates from the client
    """

    def __init__(self, 
                 algorithm_name: str,
                 model: torch.nn.Module = None, 
                 input_size: int = 0,
                 act_dim: int = 0,
                 act_limit: float = 1.0):
        """
        Args:
            algorithm_name: Name of the RL algorithm to use
            model: Optional initial model to use
            input_size: Size of the observation space
            act_dim: Size of the action space
            act_limit: Limit for continuous action values
        """
        self.algorithm_name = algorithm_name
        self._model = model
        self._lock = threading.Lock()

        # Initialize trajectory buffer
        self._current_traj = RL4SysTrajectory()
        
        # Initialize client for server communication
        self.client = RL4SysClient(
            algorithm_name=algorithm_name,
            input_size=input_size,
            act_dim=act_dim,
            act_limit=act_limit
        )
        
        # Start model update thread
        self._running = True
        self._update_thread = threading.Thread(target=self._check_for_model_updates, daemon=True)
        self._update_thread.start()

    def _validate_model(self, model: torch.nn.Module) -> None:
        """Check that the model has proper interface for generating actions."""
        assert hasattr(model, 'step'), "Model must have a .step(...) method."
        # During actual usage, the model validity will be verified by usage patterns

    def _check_for_model_updates(self):
        """
        Background thread that periodically checks for model updates from the client.
        Non-blocking to the main simulation thread.
        """
        while self._running:
            try:
                # Try to get a new model (non-blocking)
                new_model = self.client.get_latest_model(timeout=0.1)
                if new_model is not None:
                    # Got a new model, update our local copy
                    with self._lock:
                        self._model = new_model
                        print("[RL4SysAgent] Updated model from client.")
                
                # Sleep briefly to avoid tight polling
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[RL4SysAgent] Error checking for model updates: {e}")
                time.sleep(1)  # Avoid tight loop on persistent errors

    def request_for_action(self, obs: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> RL4SysAction:
        """
        Produce an action from the current model. Stores the action in our local trajectory buffer.
        """
        with self._lock:
            if self._model is None:
                # If no model available, initialize with random action and empty data
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
        Non-blocking operation.
        """
        # Add the current trajectory to the client's sending queue
        self.client.add_trajectory(self._current_traj)
        print("[RL4SysAgent] Trajectory added to client's sending queue.")
        
        # Start a new trajectory
        self._current_traj = RL4SysTrajectory()

    def close(self):
        """Cleanly close connections and threads."""
        self._running = False
        
        # Wait for update thread to finish (with timeout)
        if self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
            
        # Close client connection
        self.client.close()
        print("[RL4SysAgent] Agent closed successfully.")
