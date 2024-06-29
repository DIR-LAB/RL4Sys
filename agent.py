from typing import NoReturn as Never

import time
import torch

from trajectory import RL4SysTrajectory
from action import RL4SysAction

import zmq
import threading

import os
import json
"""Import and load RL4Sys/config.json server configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
train_server = {}
load_model_path = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        train_server = config['server']
        train_server = train_server['training_server']
        load_model_path = config['model_paths']
        load_model_path = os.path.join(os.path.dirname(__file__), load_model_path['load_model'])
except (FileNotFoundError, KeyError):
    print(f"[RL4SysAgent: Failed to load configuration from {CONFIG_PATH}, loading defaults.]")
    train_server = {
        'prefix': 'tcp://',
        'host': '*',
        'port': ":5556"
    }
    load_model_path = os.path.join(os.path.dirname(__file__), 'models/model.pth')

class RL4SysAgent:
    """RL model for use in environment scripts.

    Listens for updated models on the network asynchronously.
    Updated models are saved to ./model.pth and then loaded for use.

    Initialization will not complete until a model is received over the network.

    Attributes:
        _lock (threading.Lock): to be acquired anytime self._model is accessed.
        port (int): TCP port on which to listen for updated models from training server.

    """
    def __init__(self, port: int = train_server['port']):
        self._lock = threading.Lock()
        self.port = port

        self._listen_thread = threading.Thread(target=self._loop_for_updated_model)
        self._listen_thread.daemon = True
        self._listen_thread.start()

        self._model = None # TODO type hint _model? because of _model.step, we would need all kernels to inherit from class myClass(nn.Module) with a .step() function
        self._current_traj = RL4SysTrajectory()

        # Receive one model to initialize
        while True:
            if self._model is None:
                time.sleep(1)
            else:
                break
        
        print("[RLSysAgent] Model Initialized")
        
    
    def request_for_action(self, obs: torch.Tensor, mask: torch.Tensor, reward) -> RL4SysAction:
        """Produce action based on trained model and given observation.

        Automatically records action to trajectory.

        Mask should contain 1 for all actions which are able to be chosen, and 0 for disabled.
        For example, if kernel_size is 6 but only 4 actions are available in this observation, mask unused spots:
            [1, 1, 1, 1, 0, 0]

        Args:
            obs: flattened observation. Should have shape (kernel_size, kernel_dim).
            mask: observation mask. Should have shape (kernel_size).
            reward: reward of the previous action (action which led to state corresponding to obs).
        Returns:
            Selected action in an RL4SysAction object.

        """
        with self._lock:
            assert self._model is not None

            a, data = self._model.step(torch.as_tensor(obs, dtype=torch.float32), mask.reshape(1, -1))

            r4sa = RL4SysAction(obs, a, mask, reward, data, done=False)
            self._current_traj.add_action(r4sa)

            return r4sa
        
    def flag_last_action(self, reward: int) -> None:
        """Mark end of trajectory.

        Triggers sending trajectory to training server.

        Args:
            reward: reward of the previous (and final) action.
        Returns:
            Selected action in an RL4SysAction object.

        """
        r4sa = RL4SysAction(None, None, None, reward, None, True)
        self._current_traj.add_action(r4sa) # triggers send to training server, clear local trajectory

    def _loop_for_updated_model(self) -> Never:
        """Listen on network for new model.

        Asynchronous from rest of agent by running in a seperate thread.

        """
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        address = f"{train_server['prefix']}{train_server['host']}{self.port}"
        socket.bind(address)

        while True:
            # Receive the bytes and write to a file
            model_bytes = socket.recv()
            print("[RLSysAgent - loop_for_updated_model] receives the model")

            with open(load_model_path, 'wb') as f:
                f.write(model_bytes)
            
            with self._lock:
                self._model = torch.load(f"{load_model_path}", map_location=torch.device('cpu'))

            print("[RLSysAgent - loop_for_updated_model] loaded the new model")

        socket.close()
        context.term()