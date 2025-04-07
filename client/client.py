import sys
import os
import threading
import time
import queue
import uuid
import collections
from pathlib import Path

import torch
import grpc
from protocol import trajectory_pb2, trajectory_pb2_grpc
from protocol.trajectory import RL4SysTrajectory

from utils.util import deserialize_model, serialize_action, CircularTrajectoryBuffer, DEFAULT_BUFFER_SIZE
from utils.conf_loader import ConfigLoader

from algorithms.PPO.kernel import RLActorCritic
from algorithms.DQN.kernel import DeepQNetwork
from algorithms.DDPG.kernel import Actor
from algorithms.RPO.kernel import RPOActorCritic

config_loader = ConfigLoader()
TRAINING_SERVER_ADDRESS = "localhost:50051"  # or from config

class RL4SysClient:
    """
    Asynchronous gRPC-based Client that:
      - Runs in a separate thread to avoid blocking the main simulation
      - Handles all communication with the training server
      - Maintains thread-safe queues for model and trajectory exchange
      - Receives model updates via streaming RPC
    """

    def __init__(self, 
                 algorithm_name: str,
                 input_size: int,
                 act_dim: int,
                 act_limit: float = 1.0,
                 server_address: str = TRAINING_SERVER_ADDRESS,
                 trajectory_buffer_size: int = DEFAULT_BUFFER_SIZE):
        """
        Args:
            algorithm_name: Name of the RL algorithm to use
            input_size: Dimension of observation space
            act_dim: Dimension of action space
            act_limit: Limit of continuous action values
            server_address: "host:port" of the gRPC training server
            trajectory_buffer_size: Maximum number of trajectories to buffer
        """
        self.algorithm_name = algorithm_name
        self.server_address = server_address
        self.input_size = input_size
        self.act_dim = act_dim
        self.act_limit = act_limit
        
        # Generate a unique client ID
        self.client_id = f"client-{uuid.uuid4()}"
        
        config_loader = ConfigLoader(algorithm=algorithm_name)
        self.hyperparams = config_loader.algorithm_params
        
        # Thread-safe circular buffer for trajectories
        self.trajectory_queue = CircularTrajectoryBuffer(max_size=trajectory_buffer_size)
        
        # Thread-safe queue for models (doesn't need to be circular since we only care about the latest)
        self.model_queue = queue.Queue(maxsize=5)  # Limit to 5 pending models
        
        # Threading locks
        self._lock = threading.Lock()
        self.local_version = 0
        
        # Create debug directory
        self.debug_dir = Path('debug')
        self.debug_dir.mkdir(exist_ok=True)
        
        # Initialize gRPC connection
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = trajectory_pb2_grpc.RL4SysRouteStub(self.channel)
        
        # Start client threads
        self._running = True
        self._sender_thread = threading.Thread(target=self._trajectory_sender, daemon=True)
        self._sender_thread.start()
        
        # Initial handshake to get model
        initial_model = self._handshake_for_initial_model()
        if initial_model:
            self.model_queue.put(initial_model)
            
        # Start streaming updates thread (replaces poller thread)
        self._stream_thread = threading.Thread(target=self._stream_model_updates, daemon=True)
        self._stream_thread.start()
            
    def _handshake_for_initial_model(self):
        """
        One-time handshake with the server to get an initial model.
        Returns the deserialized model if successful, None otherwise.
        """
        print("[RL4SysClient] Handshake: requesting initial model from server...")
        req = trajectory_pb2.RequestModel(first_time=1, version=self.local_version)
        try:
            resp = self.stub.ClientPoll(req)
        except grpc.RpcError as e:
            print(f"[RL4SysClient] gRPC error during handshake: {e.details()}")
            return None

        if resp.code == 1:
            print("[RL4SysClient] Handshake successful with server.")
            self.local_version = resp.version
            if len(resp.model) > 0:
                model = self._deserialize_model_response(resp)
                print("[RL4SysClient] Received and loaded initial model from server.")
                return model
            else:
                print("[RL4SysClient] Server has no initial model yet (model bytes empty).")
        elif resp.code == 0:
            print("[RL4SysClient] Server not ready yet (code=0). No initial model.")
        else:  # code == -1 or other
            print(f"[RL4SysClient] Handshake error or server error: code={resp.code}, err={resp.error}")
        
        return None
        
    def _deserialize_model_response(self, response):
        """Helper method to deserialize model from server response"""
        model = None
        if self.algorithm_name == "DQN":
            model = DeepQNetwork(input_size=self.input_size, act_dim=self.act_dim)
            model.q_network = deserialize_model(response.model)
        elif self.algorithm_name == "PPO":
            model = RLActorCritic(input_size=self.input_size, act_dim=self.act_dim)
            model.actor = deserialize_model(response.model)
            model.critic = deserialize_model(response.model_critic)
        elif self.algorithm_name == "DDPG":
            model = Actor(input_size=self.input_size, act_dim=self.act_dim, act_limit=self.act_limit, 
                          noise_scale=self.hyperparams.get('noise_scale', 0.1))
            model = deserialize_model(response.model)
        elif self.algorithm_name == "RPO":
            model = RPOActorCritic(input_size=self.input_size, act_dim=self.act_dim, 
                                  rpo_alpha=self.hyperparams.get('rpo_alpha', 0.1))
            model.actor = deserialize_model(response.model)
            model.critic = deserialize_model(response.model_critic)
        
        return model
    
    def _stream_model_updates(self):
        """
        Thread that maintains a streaming connection to the server
        for receiving model updates as they become available.
        """
        retry_count = 0
        max_retries = 10
        retry_delay = 2  # seconds
        
        while self._running:
            try:
                print(f"[RL4SysClient] Registering for model update stream (version {self.local_version})")
                
                # Register for updates with our client ID and current model version
                register_request = trajectory_pb2.ClientRegistration(
                    client_id=self.client_id,
                    current_version=self.local_version
                )
                
                # Start streaming connection
                update_stream = self.stub.RegisterForUpdates(register_request)
                
                # Process updates as they arrive
                for update in update_stream:
                    if update.code == 1 and len(update.model) > 0:
                        # Valid model update received
                        with self._lock:
                            if update.version > self.local_version:
                                # Newer model - deserialize and queue it
                                model = self._deserialize_model_response(update)
                                self.local_version = update.version
                                
                                # If model queue is full, remove oldest to make room
                                try:
                                    self.model_queue.put_nowait(model)
                                    print(f"[RL4SysClient] Received model update (version {update.version})")
                                except queue.Full:
                                    # Queue is full, get and discard an old model to make room
                                    try:
                                        self.model_queue.get_nowait()
                                        self.model_queue.put_nowait(model)
                                        print(f"[RL4SysClient] Model queue full, discarded oldest model to add version {update.version}")
                                    except (queue.Empty, queue.Full):
                                        print(f"[RL4SysClient] Failed to add model version {update.version} to queue")
                            else:
                                print(f"[RL4SysClient] Ignoring model update with version {update.version} (already have {self.local_version})")
                    elif update.code == -1:
                        print(f"[RL4SysClient] Server error: {update.error}")
                
                # If we get here, the stream ended unexpectedly
                print("[RL4SysClient] Model update stream ended")
                retry_count += 1
                
            except grpc.RpcError as e:
                print(f"[RL4SysClient] gRPC error in model stream: {e.details()}")
                retry_count += 1
            except Exception as e:
                print(f"[RL4SysClient] Error in model stream: {e}")
                retry_count += 1
                
            # If we've reached max retries, give up
            if retry_count >= max_retries and max_retries > 0:
                print(f"[RL4SysClient] Giving up on model stream after {retry_count} retries")
                break
                
            # Wait before reconnecting
            if self._running:
                time.sleep(retry_delay)
    
    def _trajectory_sender(self):
        """
        Background thread that monitors the trajectory queue and sends
        trajectories to the server when available.
        """
        while self._running:
            try:
                # Try to get a trajectory from the circular buffer
                try:
                    trajectory = self.trajectory_queue.get(timeout=0.5)
                except queue.Empty:
                    # No trajectory available, continue polling
                    continue
                
                # We got a trajectory, send it to the server
                action_msgs = []
                for action in trajectory.actions:
                    action_proto = serialize_action(action)
                    action_msgs.append(action_proto)

                action_list = trajectory_pb2.RL4SysActionList(actions=action_msgs)

                try:
                    response = self.stub.SendActions(action_list)
                    if response.code == 1:
                        print(f"[RL4SysClient] Successfully sent trajectory: {response.message}")
                    else:
                        print(f"[RL4SysClient] Server rejected trajectory: {response.message}")
                except grpc.RpcError as e:
                    print(f"[RL4SysClient] gRPC error sending trajectory: {e.details()}")
                    
                # Mark the task as done
                self.trajectory_queue.task_done()
                
                # Periodically log buffer stats
                if self.trajectory_queue.total_removed % 10 == 0:
                    stats = self.trajectory_queue.stats()
                    print(f"[RL4SysClient] Trajectory buffer stats: {stats}")
                
            except Exception as e:
                print(f"[RL4SysClient] Error in trajectory sender: {e}")
                time.sleep(1)  # Avoid tight loop on persistent errors
                
    def add_trajectory(self, trajectory):
        """
        Add a trajectory to the circular buffer.
        Non-blocking operation, may discard oldest trajectory if buffer is full.
        """
        self.trajectory_queue.put(trajectory)
        
    def get_latest_model(self, timeout=0.1):
        """
        Try to get the latest model from the queue.
        Returns None if no new model is available.
        """
        try:
            return self.model_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        
    def get_buffer_stats(self):
        """Return statistics about the trajectory buffer usage"""
        return self.trajectory_queue.stats()
        
    def close(self):
        """Cleanly close the client connections and threads."""
        self._running = False
        
        # Wait for threads to finish (with timeout)
        if self._sender_thread.is_alive():
            self._sender_thread.join(timeout=1.0)
        if self._stream_thread.is_alive():
            self._stream_thread.join(timeout=1.0)
            
        self.channel.close()
        print("[RL4SysClient] Closed gRPC channel and stopped threads.")
