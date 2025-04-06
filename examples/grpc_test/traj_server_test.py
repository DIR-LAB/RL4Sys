import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import grpc
from concurrent import futures
from protocol import trajectory_pb2
from protocol import trajectory_pb2_grpc
import threading
import io
import torch
import numpy as np
from utils.util import deserialize_tensor, serialize_model

class LunarLanderModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(8, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4)
        )
    
    def forward(self, x):
        return self.network(x)
    
    def step(self, obs, mask):
        """Required method for the agent interface"""
        with torch.no_grad():
            if obs is not None:
                logits = self.forward(obs)
                # Simple argmax policy
                action = torch.argmax(logits).item()
                return np.array(action), {"logits": logits.numpy()}
            return np.array(0), {}

class RL4SysRouteServicer(trajectory_pb2_grpc.RL4SysRouteServicer):
    def __init__(self):
        self.model_ready = True  # Changed to True initially
        self.lock = threading.Lock()
        self.version = 0
        
        # Create and serialize initial model
        initial_model = LunarLanderModel()  # Using our custom model
        self.trained_model = serialize_model(initial_model)
        self.error_message = None

    def SendActions(self, request, context):
        print(f"Received {len(request.actions)} actions for training.")
        for action in request.actions:
            obs_tensor = deserialize_tensor(action.obs)
            action_tensor = deserialize_tensor(action.action)
            mask_tensor = deserialize_tensor(action.mask)
            print("Deserialized Action Fields:")
            print(f"  obs: {obs_tensor}, {type(obs_tensor)}")
            print(f"  action: {action_tensor}, {type(action_tensor)}")
            print(f"  mask: {mask_tensor}, {type(mask_tensor)}")
            print(f"  reward: {action.reward}")
            print(f"  data: {action.data}")
            print(f"  done: {action.done}")
            print(f"  reward_update_flag: {action.reward_update_flag}")

        def simulate_training():
            try:
                time.sleep(5)  # Simulate training time
                model = LunarLanderModel()  # Using our custom model
                with self.lock:  # Protect shared state
                    self.trained_model = serialize_model(model)
                    self.model_ready = True
                    self.version += 1
            except Exception as e:
                with self.lock:
                    self.model_ready = -1
                    self.error_message = str(e)

        self.model_ready = False
        threading.Thread(target=simulate_training).start()
        return trajectory_pb2.ActionResponse(code=1, message="Training started successfully.")

    def ClientPoll(self, request, context):
        print(f'[Client Poll] Received client poll request... (first_time={request.first_time}, version={request.version})')
        with self.lock:  # Protect shared state
            if request.first_time == 1:  # Initial handshake
                print('[Client Poll] First time request - sending initial model')
                return trajectory_pb2.RL4SysModel(
                    code=1, 
                    model=self.trained_model, 
                    error="",
                    version=self.version
                )
            elif self.model_ready == True:
                print('[Client Poll] Model is ready. Sending Model')
                return trajectory_pb2.RL4SysModel(
                    code=1, 
                    model=self.trained_model, 
                    error="",
                    version=self.version
                )
            elif self.model_ready == -1:
                print('[Client Poll] Error:', self.error_message)
                return trajectory_pb2.RL4SysModel(
                    code=-1, 
                    model=b"", 
                    error=self.error_message,
                    version=self.version
                )
            else:
                print('[Client Poll] Model is still training...')
                return trajectory_pb2.RL4SysModel(
                    code=0, 
                    model=b"", 
                    error="Model is still training.",
                    version=self.version
                )
            
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    trajectory_pb2_grpc.add_RL4SysRouteServicer_to_server(RL4SysRouteServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started, listening on port 50051.")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
