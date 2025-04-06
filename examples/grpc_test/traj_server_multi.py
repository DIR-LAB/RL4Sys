import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import re
import grpc
from concurrent import futures
from protocol import trajectory_pb2
from protocol import trajectory_pb2_grpc
import threading
import io
import torch
from collections import OrderedDict
from utils.util import deserialize_tensor


# Class to hold individual client state
class ClientState:
    def __init__(self, client_id):
        self.client_id = client_id
        self.model_ready = 0
        self.trained_model_path = os.path.join(os.path.dirname(__file__), 'models', f"{client_id}_model.pth")
        self.trained_model = None
        self.error_message = None
        self.last_interaction = time.time()  # Timestamp of the last interaction
        self.lock = threading.Lock()

    def save_model(self, model_data):
        """Save model to persistent storage."""
        os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)  # Ensure the directory exists
        with open(self.trained_model_path, "wb") as f:
            f.write(model_data)

    def load_model(self):
        """Load model from persistent storage."""
        with open(self.trained_model_path, "rb") as f:
            return f.read()


# gRPC server class to manage multiple clients
class RL4SysRouteServicer(trajectory_pb2_grpc.RL4SysRouteServicer):
    def __init__(self, max_clients=100, cleanup_interval=300, idle_timeout=30):
        self.clients = {}  # Dictionary to store client states
        self.max_clients = max_clients  # Maximum number of clients to retain
        self.cleanup_interval = cleanup_interval  # Time between cleanup runs (seconds)
        self.idle_timeout = idle_timeout  # Maximum idle time before removing a client
        self.lock = threading.Lock()
        threading.Thread(target=self._cleanup_inactive_clients, daemon=True).start()

    def _sanitize_client_id(self, client_id):
        """Sanitize the client ID for safe file naming."""
        # Replace any non-alphanumeric characters with underscores
        return re.sub(r'[^a-zA-Z0-9]', '_', client_id)
    
    def _get_client_id(self, context):
        """Extract a unique identifier for the client."""
        client_ip = context.peer()  # Get client connection details
        client_ip = self._sanitize_client_id(client_ip)
        with self.lock:
            if client_ip not in self.clients:
                # If the number of clients exceeds the limit, remove the least recently used (LRU)
                if len(self.clients) >= self.max_clients:
                    print("Try to add client into server buffer but buffer is full. Do nothing")
                    return -1

                # Add a new client to the dictionary
                print(f"New client connected: {client_ip}")
                self.clients[client_ip] = ClientState(client_id=client_ip)

            # Update the last interaction timestamp
            self.clients[client_ip].last_interaction = time.time()
        return client_ip

    def _cleanup_inactive_clients(self):
        """Periodically remove inactive clients."""
        while True:
            time.sleep(self.cleanup_interval)
            with self.lock:
                current_time = time.time()
                inactive_clients = [
                    client_id for client_id, state in self.clients.items()
                    if current_time - state.last_interaction > self.idle_timeout
                ]
                for client_id in inactive_clients:
                    print(f"Removing inactive client: {client_id}")
                    del self.clients[client_id]

    def _see_actions(self, actions):
        """This function is DEBUG only function. Traverse through all trajectory"""
        # Debug only
        for action in actions:
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

    def SendActions(self, request, context):
        client_id = self._get_client_id(context)
        if client_id == -1:
            return trajectory_pb2.ActionResponse(code=0, message=f"Training server clients serving buffer are full. Try again later.")
        
        client_state = self.clients[client_id]
        print(f"Received {len(request.actions)} actions for training from client {client_id}.")

        self._see_actions(request.actions)
        

        # Simulate model training in a separate thread
        def simulate_training():
            try:
                time.sleep(5)  # Simulate training
                model = torch.nn.Linear(10, 5)
                model_buffer = io.BytesIO()
                torch.save(model.state_dict(), model_buffer)
                with client_state.lock:
                    model_param = model_buffer.getvalue()
                    client_state.trained_model = model_param
                    client_state.save_model(model_param)  # Save the model to persistent storage
                    client_state.model_ready = 1
            except Exception as e:
                with client_state.lock:
                    print(f'Server Error: {str(e)}')
                    client_state.model_ready = -1
                    client_state.error_message = str(e)

        # clean model_ready before start 
        client_state.model_ready = 0
        threading.Thread(target=simulate_training).start()
        return trajectory_pb2.ActionResponse(code=1, message=f"Training started successfully for client {client_id}.")

    def ClientPoll(self, request, context):
        """
        parameters:
            request: Empty. Client don't send any info to server on this request
            context: gRPC transmission context
        Return:
            a RL4SysModel proto3 obj. include three fields:
            code: 1, 0, -1. 1 means all good. 0 means still training. -1 means error.
        """
        # Server received a poll from unknown server
        client_ip = context.peer()  # Get client connection details
        client_ip = self._sanitize_client_id(client_ip)
        if client_ip not in self.clients:
            print(f"[Client Poll] Error for client {client_id}: Client has not start training yet but send a poll request instead.")
            return trajectory_pb2.RL4SysModel(code=-1, model=b"", error=client_state.error_message)

        # Server client as usual        
        client_id = self._get_client_id(context)
        if client_id == -1:
            return trajectory_pb2.ActionResponse(code=0, message=f"Training server clients serving buffer are full. Try again later.")
        
        client_state = self.clients[client_id]
        print(f"[Client Poll] Received poll request from client {client_id}...")
        done = False
        timeout = 0
        while not done:
            if timeout >= self.idle_timeout:
                return trajectory_pb2.RL4SysModel(code=0, model=b"", error="Model is still training.")


            with client_state.lock:
                # Initial handshake
                if request.first_time == 1:
                    print(f"Client model version {request.version}, Server model version ") # TODO model need to have version scheme
                    print(f"[Client Poll] Handshake initiated by client {client_id}.")
                    return trajectory_pb2.RL4SysModel(code=1, model=b"", version=0, error="Handshake successful.")

                if client_state.model_ready == 1:
                    print(f"[Client Poll] Model is ready for client {client_id}. Sending model.")
                    model_data = client_state.trained_model
                    return trajectory_pb2.RL4SysModel(code=1, model=model_data, error="")
                elif client_state.model_ready == -1:
                    print(f"[Client Poll] Error for client {client_id}: {client_state.error_message}")
                    return trajectory_pb2.RL4SysModel(code=-1, model=b"", error=client_state.error_message)
            
            time.sleep(1)
            timeout += 1


# Server startup function
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    trajectory_pb2_grpc.add_RL4SysRouteServicer_to_server(RL4SysRouteServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started, listening on port 50051.")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
