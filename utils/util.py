import io
import torch
from torch import nn
from protocol.action import RL4SysAction
from protocol import trajectory_pb2
import numpy as np
import threading
import time
import queue
import collections

# for gRPC transfering. Torch tensors will be conver into bytes
def serialize_tensor(tensor):
    """Serialize a PyTorch tensor to bytes."""
    if tensor is None:
        return b"None"
    if isinstance(tensor, torch.Tensor):
        return tensor.numpy().tobytes()
    if isinstance(tensor, np.ndarray):
        return tensor.tobytes()
    # For scalar values
    return np.array([tensor]).tobytes()

def deserialize_tensor(tensor_bytes):
    """Deserialize bytes back to a PyTorch tensor."""
    if tensor_bytes == b"None":
        return None
    try:
        # Convert bytes back to numpy array
        numpy_array = np.frombuffer(tensor_bytes, dtype=np.float32)
        # Convert numpy array to torch tensor
        return torch.from_numpy(numpy_array)
    except Exception as e:
        print(f"Error deserializing tensor: {e}")
        return None

def serialize_action(action: RL4SysAction) -> trajectory_pb2.RL4SysAction:
    """
    Serialize an RL4SysAction object into a protobuf message, 
    and store reward as bytes (float->bytes).
    """
    # 1. Serialize the action
    action_bytes = b"None"
    if action.act is not None:
        if isinstance(action.act, torch.Tensor):
            action_bytes = action.act.detach().cpu().numpy().astype(np.float32).tobytes()
        elif isinstance(action.act, np.ndarray):
            action_bytes = action.act.astype(np.float32).tobytes()
        else:
            # For scalar values (int or float), convert to float32 single-element array
            action_bytes = np.array([float(action.act)], dtype=np.float32).tobytes()

    # 2. Serialize the reward as bytes
    if action.rew is not None:
        reward_bytes = np.array([float(action.rew)], dtype=np.float32).tobytes()
    else:
        reward_bytes = b"None"

    # 3. Serialize data
    serialized_data = {}
    if action.data:
        for k, v in action.data.items():
            if isinstance(v, torch.Tensor):
                serialized_data[str(k)] = v.detach().cpu().numpy().astype(np.float32).tobytes()
            elif isinstance(v, np.ndarray):
                serialized_data[str(k)] = v.astype(np.float32).tobytes()
            else:
                serialized_data[str(k)] = np.array([float(v)], dtype=np.float32).tobytes()

    # 4. Build the protobuf
    action_proto = trajectory_pb2.RL4SysAction(
        obs=serialize_tensor(action.obs) if action.obs is not None else b"None",
        action=action_bytes,
        mask=serialize_tensor(action.mask) if action.mask is not None else b"None",
        reward=reward_bytes,  # store reward as bytes
        data=serialized_data,
        done=action.done if action.done is not None else False,
        reward_update_flag=action.reward_update_flag if action.reward_update_flag is not None else False,
    )
    return action_proto

def deserialize_action(action):
    """
    Deserialize a Protobuf message into an RL4SysAction object, 
    turning bytes->float for reward.
    """
    # 1. Deserialize the action
    action_bytes = action.action
    if action_bytes == b"None":
        action_value = None
    else:
        try:
            arr = np.frombuffer(action_bytes, dtype=np.float32)
            action_value = float(arr[0]) if len(arr) == 1 else arr
        except Exception:
            print(f"Warning: Could not deserialize action: {action_bytes}")
            action_value = None

    # 2. Deserialize the reward (bytes->float)
    reward_bytes = action.reward
    if reward_bytes == b"None":
        rew_value = None
    else:
        try:
            arr = np.frombuffer(reward_bytes, dtype=np.float32)
            rew_value = float(arr[0]) if len(arr) == 1 else arr
        except Exception:
            print(f"Warning: Could not deserialize reward bytes: {reward_bytes}")
            rew_value = 0.0

    # 3. Deserialize data into dictionary
    deserialized_data = {}
    for k, v in action.data.items():
        try:
            arr = np.frombuffer(v, dtype=np.float32)
            deserialized_data[k] = float(arr[0]) if len(arr) == 1 else arr
        except:
            try:
                deserialized_data[k] = v.decode("utf-8")
            except:
                print(f"Warning: Could not deserialize data for key {k}")
                deserialized_data[k] = None

    # 4. Rebuild the RL4SysAction
    return RL4SysAction(
        obs=deserialize_tensor(action.obs),
        action=action_value,
        mask=deserialize_tensor(action.mask),
        reward=rew_value,
        data=deserialized_data,
        done=action.done
    )

def serialize_model(model: nn.Module) -> bytes:
    """
    Serializes the full PyTorch model object into bytes (including class structure).
    The client must have the same class definition(s) available on load.
    """
    buffer = io.BytesIO()
    # This pickles the entire model, including its class definitions (by reference)
    torch.save(model, buffer)
    buffer.seek(0)
    return buffer.read()

def deserialize_model(raw_bytes: bytes) -> nn.Module:
    """
    Deserializes the full PyTorch model from raw bytes.
    Requires that the exact model class definition be available locally.
    """
    buffer = io.BytesIO(raw_bytes)
    model = torch.load(buffer, map_location='cpu')
    return model

# Default maximum trajectories to buffer
DEFAULT_BUFFER_SIZE = 10

class CircularTrajectoryBuffer:
    """
    A thread-safe circular buffer for trajectories.
    When the buffer is full, the oldest trajectory is overwritten.
    """
    def __init__(self, max_size=DEFAULT_BUFFER_SIZE):
        self.max_size = max_size
        self.buffer = collections.deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.total_added = 0
        self.total_removed = 0
        self.total_discarded = 0
    
    def put(self, trajectory):
        """Add a trajectory to the buffer, potentially discarding the oldest one"""
        with self.lock:
            if len(self.buffer) >= self.max_size:
                # Buffer is full, oldest trajectory will be automatically discarded
                self.total_discarded += 1
                print(f"[CircularBuffer] Buffer full, discarding oldest trajectory (total discarded: {self.total_discarded})")
            
            self.buffer.append(trajectory)
            self.total_added += 1
            
            
    
    def get(self, block=True, timeout=None):
        """Get the next trajectory from the buffer"""
        with self.lock:
            if not self.buffer:
                if block:
                    # If blocking is requested, we need to release the lock and wait
                    start_time = time.time()
                    while not self.buffer:
                        self.lock.release()
                        time.sleep(0.1)
                        self.lock.acquire()
                        if timeout is not None and time.time() - start_time > timeout:
                            raise queue.Empty("Timeout waiting for trajectory")
                else:
                    raise queue.Empty("No trajectories available")
            
            trajectory = self.buffer.popleft()
            self.total_removed += 1
            return trajectory
    
    def get_all(self):
        """Get all trajectories from the buffer and clear it"""
        with self.lock:
            trajectories = list(self.buffer)
            self.total_removed += len(trajectories)
            self.buffer.clear()
            return trajectories
    
    def task_done(self):
        """Mark task as done (compatibility with Queue interface)"""
        pass
    
    def qsize(self):
        """Return the approximate size of the buffer"""
        with self.lock:
            return len(self.buffer)
    
    def empty(self):
        """Return True if the buffer is empty, False otherwise"""
        with self.lock:
            return len(self.buffer) == 0
    
    def full(self):
        """Return True if the buffer is full, False otherwise"""
        with self.lock:
            return len(self.buffer) >= self.max_size
    
    def clear(self):
        """Clear all trajectories from the buffer"""
        with self.lock:
            self.buffer.clear()
    
    def stats(self):
        """Return statistics about the buffer usage"""
        with self.lock:
            return {
                "current_size": len(self.buffer),
                "max_size": self.max_size,
                "total_added": self.total_added,
                "total_removed": self.total_removed,
                "total_discarded": self.total_discarded
            }


