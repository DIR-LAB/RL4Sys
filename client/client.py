import sys
import os
import threading
import time
import queue
import uuid
import collections
import random  # Add random for trajectory sampling
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
      - Can directly update agent's model when streaming updates
      - Collects trajectories in a buffer and sends them in batches when a model update is received
      - Can randomly sample a configurable number of trajectories to send, rather than sending all
    """

    def __init__(self, 
                 algorithm_name: str,
                 input_size: int,
                 act_dim: int,
                 act_limit: float = 1.0,
                 server_address: str = TRAINING_SERVER_ADDRESS,
                 trajectory_buffer_size: int = DEFAULT_BUFFER_SIZE,
                 agent_reference = None,
                 min_trajectories_to_send: int = 1,
                 max_trajectories_to_send: int = 10,
                 verbose: bool = True):
        """
        Args:
            algorithm_name: Name of the RL algorithm to use
            input_size: Dimension of observation space
            act_dim: Dimension of action space
            act_limit: Limit of continuous action values
            server_address: "host:port" of the gRPC training server
            trajectory_buffer_size: Maximum number of trajectories to buffer
            agent_reference: Reference to the agent for direct model updates
            min_trajectories_to_send: Minimum number of trajectories to collect before sending
            max_trajectories_to_send: Maximum number of trajectories to send in one batch (samples if more are available)
            verbose: Whether to print status messages
        """
        self.algorithm_name = algorithm_name
        self.server_address = server_address
        self.input_size = input_size
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.agent_reference = agent_reference
        self.min_trajectories_to_send = min_trajectories_to_send
        self.max_trajectories_to_send = max_trajectories_to_send
        self.verbose = verbose
        
        # Generate a unique client ID
        self.client_id = f"client-{uuid.uuid4()}"
        
        config_loader = ConfigLoader(algorithm=algorithm_name)
        self.hyperparams = config_loader.algorithm_params
        
        # Thread-safe circular buffer for trajectories
        self.trajectory_queue = CircularTrajectoryBuffer(max_size=trajectory_buffer_size)
        
        # Thread-safe queue for models (doesn't need to be circular since we only care about the latest)
        # We'll keep this queue for compatibility, but direct updates will be preferred when agent_reference is provided
        self.model_queue = queue.Queue(maxsize=5)  # Limit to 5 pending models
        
        # Threading locks
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()  # Lock for sending operations
        self.local_version = 0
        
        # Flag to ensure we send trajectories at least once before receiving model updates
        self.first_send_done = False
        
        # Create debug directory
        self.debug_dir = Path('debug')
        self.debug_dir.mkdir(exist_ok=True)
        
        # Initialize gRPC connection
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = trajectory_pb2_grpc.RL4SysRouteStub(self.channel)
        
        # Start client threads
        self._running = True
        
        # Initial handshake to get model
        initial_model = self._handshake_for_initial_model()
        if initial_model:
            # If we have an agent reference, update the agent's model directly
            if self.agent_reference:
                with self.agent_reference._lock:
                    self.agent_reference._model = initial_model
                    if self.verbose:
                        print("[RL4SysClient] Updated agent model directly from initial handshake.")
            else:
                # Otherwise, use the queue as before
                self.model_queue.put(initial_model)
            
        # Start streaming updates thread
        self._stream_thread = threading.Thread(target=self._stream_model_updates, daemon=True)
        self._stream_thread.start()
            
    def _handshake_for_initial_model(self):
        """
        One-time handshake with the server to get an initial model.
        Returns the deserialized model if successful, None otherwise.
        """
        if self.verbose:
            print("[RL4SysClient] Handshake: requesting initial model from server...")
        req = trajectory_pb2.RequestModel(first_time=1, version=self.local_version)
        try:
            resp = self.stub.ClientPoll(req)
        except grpc.RpcError as e:
            if self.verbose:
                print(f"[RL4SysClient] gRPC error during handshake: {e.details()}")
            return None

        if resp.code == 1:
            if self.verbose:
                print("[RL4SysClient] Handshake successful with server.")
            self.local_version = resp.version
            if len(resp.model) > 0:
                model = self._deserialize_model_response(resp)
                if self.verbose:
                    print("[RL4SysClient] Received and loaded initial model from server.")
                return model
            else:
                if self.verbose:
                    print("[RL4SysClient] Server has no initial model yet (model bytes empty).")
        elif resp.code == 0:
            if self.verbose:
                print("[RL4SysClient] Server not ready yet (code=0). No initial model.")
        else:  # code == -1 or other
            if self.verbose:
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
        Now can directly update the agent's model if agent_reference is provided.
        Also triggers trajectory batch sending when a model update is received.
        """
        retry_count = 0
        max_retries = 10
        retry_delay = 2  # seconds
        
        while self._running:
            try:
                if self.verbose:
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
                                # Newer model - deserialize it
                                model = self._deserialize_model_response(update)
                                self.local_version = update.version
                                
                                # If we have an agent reference, update the agent's model directly
                                if self.agent_reference:
                                    with self.agent_reference._lock:
                                        self.agent_reference._model = model
                                        if self.verbose:
                                            print(f"[RL4SysClient] Directly updated agent model to version {update.version}")
                                else:
                                    # Otherwise, use the queue as before
                                    try:
                                        self.model_queue.put_nowait(model)
                                        if self.verbose:
                                            print(f"[RL4SysClient] Received model update (version {update.version})")
                                    except queue.Full:
                                        # Queue is full, get and discard an old model to make room
                                        try:
                                            self.model_queue.get_nowait()
                                            self.model_queue.put_nowait(model)
                                            if self.verbose:
                                                print(f"[RL4SysClient] Model queue full, discarded oldest model to add version {update.version}")
                                        except (queue.Empty, queue.Full):
                                            if self.verbose:
                                                print(f"[RL4SysClient] Failed to add model version {update.version} to queue")
                                
                                # After receiving a model update, send any collected trajectories
                                # and clear the trajectory buffer
                                if not self.trajectory_queue.empty():
                                    if self.verbose:
                                        print(f"[RL4SysClient] Sending trajectories after model update (version {update.version})")
                                    self.send_trajectories_batch()
                            else:
                                if self.verbose:
                                    print(f"[RL4SysClient] Ignoring model update with version {update.version} (already have {self.local_version})")
                    elif update.code == -1:
                        if self.verbose:
                            print(f"[RL4SysClient] Server error: {update.error}")
                
                # If we get here, the stream ended unexpectedly
                if self.verbose:
                    print("[RL4SysClient] Model update stream ended")
                retry_count += 1
                
            except grpc.RpcError as e:
                if self.verbose:
                    print(f"[RL4SysClient] gRPC error in model stream: {e.details()}")
                retry_count += 1
            except Exception as e:
                if self.verbose:
                    print(f"[RL4SysClient] Error in model stream: {e}")
                retry_count += 1
                
            # If we've reached max retries, give up
            if retry_count >= max_retries and max_retries > 0:
                if self.verbose:
                    print(f"[RL4SysClient] Giving up on model stream after {retry_count} retries")
                break
                
            # Wait before reconnecting
            if self._running:
                time.sleep(retry_delay)
    
    def send_trajectories_batch(self):
        """
        Collects buffered trajectories and sends them as a batch.
        Creates a one-time thread to handle the sending so it doesn't block.
        After sending, clears old trajectories from the buffer.
        """
        # Use a lock to ensure only one send operation happens at a time
        if not self._send_lock.acquire(blocking=False):
            if self.verbose:
                print("[RL4SysClient] Another send operation is in progress, skipping this one")
            return
        
        try:
            # Check if we have enough trajectories to send
            if self.trajectory_queue.qsize() < self.min_trajectories_to_send and self.first_send_done:
                if self.verbose:
                    print(f"[RL4SysClient] Not enough trajectories to send (have {self.trajectory_queue.qsize()}, need {self.min_trajectories_to_send})")
                return
            
            # Get all trajectories from the buffer - this clears the buffer
            trajectories = self.trajectory_queue.get_all()
            
            if not trajectories:
                if self.verbose:
                    print("[RL4SysClient] No trajectories to send")
                return
                
            if self.verbose:
                print(f"[RL4SysClient] Preparing to send batch of trajectories (collected {len(trajectories)})")
            
            # Create and start a dedicated thread for sending
            send_thread = threading.Thread(
                target=self._send_trajectories_thread, 
                args=(trajectories,),
                daemon=True
            )
            send_thread.start()
            
            # Mark that we've done at least one send operation
            self.first_send_done = True
            
        finally:
            # Release the lock
            self._send_lock.release()
    
    def _send_trajectories_thread(self, trajectories):
        """
        Thread function that actually sends the trajectory batch to the server.
        This runs in a dedicated thread so it doesn't block the main process.
        If there are more trajectories than max_trajectories_to_send, randomly sample that many.
        """
        try:
            # Sample trajectories if we have more than the maximum to send
            if len(trajectories) > self.max_trajectories_to_send:
                if self.verbose:
                    print(f"[RL4SysClient] Sampling {self.max_trajectories_to_send} trajectories from {len(trajectories)} available")
                sampled_trajectories = random.sample(trajectories, self.max_trajectories_to_send)
            else:
                # Otherwise send all available trajectories
                sampled_trajectories = trajectories
                
            # Create a batch message
            batch = trajectory_pb2.RL4SysTrajectoryBatch(batch_size=len(sampled_trajectories))
            
            # Add each trajectory to the batch
            total_actions = 0
            for trajectory in sampled_trajectories:
                action_msgs = []
                for action in trajectory.actions:
                    action_proto = serialize_action(action)
                    action_msgs.append(action_proto)
                    total_actions += 1
                
                action_list = trajectory_pb2.RL4SysActionList(actions=action_msgs)
                batch.trajectories.append(action_list)
            
            # Send the batch to the server
            try:
                response = self.stub.SendTrajectoryBatch(batch)
                if response.code == 1:
                    if self.verbose:
                        print(f"[RL4SysClient] Successfully sent batch of {len(sampled_trajectories)} trajectories ({total_actions} actions): {response.message}")
                else:
                    if self.verbose:
                        print(f"[RL4SysClient] Server rejected trajectory batch: {response.message}")
            except grpc.RpcError as e:
                if self.verbose:
                    print(f"[RL4SysClient] gRPC error sending trajectory batch: {e.details()}")
                
        except Exception as e:
            if self.verbose:
                print(f"[RL4SysClient] Error in trajectory batch send thread: {e}")
    
    def add_trajectory(self, trajectory):
        """
        Add a trajectory to the circular buffer.
        If this is the first trajectory and we haven't sent any yet,
        or if we have accumulated enough trajectories, trigger a send.
        """
        self.trajectory_queue.put(trajectory)
        if self.verbose:
            print(f"[RL4SysClient] Added trajectory to buffer (size: {self.trajectory_queue.qsize()})")
        
        # Check if we should send trajectories now
        if not self.first_send_done or self.trajectory_queue.qsize() >= self.min_trajectories_to_send:
            self.send_trajectories_batch()
        
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
        
        # Send any remaining trajectories before shutting down
        if not self.trajectory_queue.empty():
            if self.verbose:
                print("[RL4SysClient] Sending remaining trajectories before shutdown")
            self.send_trajectories_batch()
        
        # Wait for stream thread to finish (with timeout)
        if self._stream_thread.is_alive():
            self._stream_thread.join(timeout=1.0)
            
        self.channel.close()
        if self.verbose:
            print("[RL4SysClient] Closed gRPC channel and stopped threads.")
