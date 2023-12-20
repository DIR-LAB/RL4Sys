import random
import time
import numpy as np
import torch

from models.kernel_ac import RLActorCritic
from trajectory import RL4SysTrajectory
from action import RL4SysAction
from observation import RL4SysObservation

import zmq
import threading
from io import BytesIO


class RL4SysAgent:
    def __init__(self, port=5556):
        self.lock = threading.Lock()
        self.port = port

        self.listen_thread = threading.Thread(target=self.loop_for_updated_model)
        self.listen_thread.start()

        self.model = None
        self.current_traj = RL4SysTrajectory()

        # make sure we received one model first
        while True:
            if self.model is None:
                time.sleep(1)
            else:
                break
        
        print("[RLSysAgent] Model Initialized")
        
    # the reward r here is the reward from last action. 
    def request_for_action(self, obs, mask, r) -> RL4SysAction:
        with self.lock:
            assert self.model is not None
            a, v_t, logp_t = self.model.step(torch.as_tensor(obs, dtype=torch.float32), mask.reshape(1, -1))

            r4sa = RL4SysAction(obs, None, a, None, r, v_t, logp_t, False)
            self.current_traj.add_action(r4sa)

            return r4sa
        
    def flag_last_action(self, r):
        r4sa = RL4SysAction(None, None, None, None, r, None, None, True)
        self.current_traj.add_action(r4sa)

    def loop_for_updated_model(self):
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{self.port}")

        while True:
            # Receive the bytes and write to a file
            model_bytes = socket.recv()
            print("[RLSysAgent - loop_for_updated_model] receives the model")

            with open('model.pth', 'wb') as f:
                f.write(model_bytes)
            
            with self.lock:
                self.model = torch.load('model.pth', map_location=torch.device('cpu'))

            print("[RLSysAgent - loop_for_updated_model] load the new model")

        socket.close()
        context.term()
