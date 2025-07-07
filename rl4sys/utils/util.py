import io
import torch
from torch import nn
from rl4sys.common.action import RL4SysAction
from rl4sys.proto import Action
import numpy as np
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# for gRPC transfering. Torch tensors will be conver into bytes
def serialize_tensor(tensor):
    """
    Convert a tensor (torch.Tensor or np.ndarray) to bytes.
    """
    if tensor is None:
        return b"None"
    
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().astype(np.float32).tobytes()
    elif isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32).tobytes()
    else:
        # For scalar values (int or float), convert to float32 single-element array
        return np.array([float(tensor)], dtype=np.float32).tobytes()

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

def serialize_action(action: RL4SysAction) -> Action:
    """
    Serialize an RL4SysAction object into a protobuf message according to the new proto definition.
    """
    # Serialize the observation, action, reward, and mask
    obs_bytes = serialize_tensor(action.obs)
    action_bytes = serialize_tensor(action.act)
    reward_bytes = serialize_tensor(action.rew)
    mask_bytes = serialize_tensor(action.mask)
    
    # Serialize extra data
    extra_data = {}
    if action.data:
        for k, v in action.data.items():
            extra_data[str(k)] = serialize_tensor(v)
    
    # Build the protobuf message
    action_proto = Action(
        obs=obs_bytes,
        action=action_bytes,
        reward=reward_bytes,
        done=action.done if action.done is not None else False,
        mask=mask_bytes,
        extra_data=extra_data
    )
    
    return action_proto

def deserialize_action(action_proto: Action) -> RL4SysAction:
    """
    Convert a protobuf Action message back to an RL4SysAction object.
    This is typically used by the server when receiving trajectories.
    """
    # Deserialize tensors from bytes
    if action_proto.obs == b"None":
        obs = None
    else:
        obs = np.frombuffer(action_proto.obs, dtype=np.float32)
    
    if action_proto.action == b"None":
        action = None
    else:
        action = np.frombuffer(action_proto.action, dtype=np.float32)
    
    if action_proto.reward == b"None":
        reward = None
    else:
        reward = np.frombuffer(action_proto.reward, dtype=np.float32)[0]  # Get scalar value
    
    # Deserialize mask
    if action_proto.mask == b"None":
        mask = None
    else:
        mask = np.frombuffer(action_proto.mask, dtype=np.float32)
        # Convert numpy array to torch tensor to match original format
        mask = torch.from_numpy(mask)
    
    # Deserialize extra data
    data = {}
    for k, v in action_proto.extra_data.items():
        data[k] = np.frombuffer(v, dtype=np.float32)
        # If it's a single value, convert to scalar
        if len(data[k]) == 1:
            data[k] = float(data[k][0])
    
    # Create and return the RL4SysAction
    return RL4SysAction(
        obs=obs,
        action=action,
        reward=reward,
        done=action_proto.done,
        mask=mask,
        data=data
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

class StructuredLogger:
    """
    A structured logger that provides context-aware logging.
    
    This logger integrates with the centralized logging configuration
    and adds JSON-formatted context data to log records.
    """
    
    def __init__(self, name: str, debug: bool = False):
        """
        Initialize a structured logger.
        
        Args:
            name: Logger name
            debug: Enable debug logging
        """
        self.logger = logging.getLogger(name)
        self.debug_mode = debug
        
        # Don't override centralized logging configuration
        # Just use the existing logger with its configured handlers
        
        # Performance optimization: cache logging level checks
        self._debug_enabled = self.logger.isEnabledFor(logging.DEBUG)
        self._info_enabled = self.logger.isEnabledFor(logging.INFO)
    
    def _get_context(self, **kwargs):
        """
        Return a JSON string for the ``extra['context']`` field.

        NumPy scalars (and any other non-JSON types) are converted to
        something the stdlib encoder can handle.
        """
        def _default(obj):
            # Convert NumPy scalars → native Python types
            if isinstance(obj, np.generic):
                return obj.item()
            # Fallback: give up and use the string form
            return str(obj)
        return json.dumps(kwargs, default=_default)
    
    def info(self, msg, **kwargs):
        """Log an info level message with optional context."""
        if self._info_enabled:
            self.logger.info(msg, extra={'context': self._get_context(**kwargs)})
    
    def debug(self, msg, **kwargs):
        """Log a debug level message with optional context."""
        if self._debug_enabled:
            self.logger.debug(msg, extra={'context': self._get_context(**kwargs)})
    
    def error(self, msg, **kwargs):
        """Log an error level message with optional context."""
        # Error messages are always logged regardless of level
        self.logger.error(msg, extra={'context': self._get_context(**kwargs)})
    
    def warning(self, msg, **kwargs):
        """Log a warning level message with optional context."""
        self.logger.warning(msg, extra={'context': self._get_context(**kwargs)})
    
    def is_debug_enabled(self):
        """Check if debug logging is enabled to avoid expensive operations."""
        return self._debug_enabled
    
    def is_info_enabled(self):
        """Check if info logging is enabled to avoid expensive operations."""
        return self._info_enabled


