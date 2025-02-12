import io
import torch
from torch import nn
from protocol.action import RL4SysAction
from protocol import trajectory_pb2
import numpy as np

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
    """Serialize an RL4SysAction object into a protobuf message."""
    # Handle the action value which could be a tensor, ndarray, or number
    action_bytes = b"None"
    if action.act is not None:
        if isinstance(action.act, torch.Tensor):
            action_bytes = action.act.numpy().tobytes()
        elif isinstance(action.act, np.ndarray):
            action_bytes = action.act.tobytes()
        else:
            # For scalar values, convert to int and then to bytes
            action_bytes = int(action.act).to_bytes(4, byteorder='big', signed=True)

    # Convert data values to bytes if they're numpy arrays or tensors
    serialized_data = {}
    if action.data:
        for k, v in action.data.items():
            if isinstance(v, torch.Tensor):
                serialized_data[str(k)] = v.numpy().tobytes()
            elif isinstance(v, np.ndarray):
                serialized_data[str(k)] = v.tobytes()
            elif isinstance(v, (int, float)):
                serialized_data[str(k)] = np.array([v]).tobytes()
            else:
                # For other types, convert to string then to bytes
                serialized_data[str(k)] = str(v).encode('utf-8')

    # Create the protobuf message
    action_proto = trajectory_pb2.RL4SysAction(
        obs=serialize_tensor(action.obs) if action.obs is not None else b"None",
        action=action_bytes,
        mask=serialize_tensor(action.mask) if action.mask is not None else b"None",
        reward=int(action.rew) if action.rew is not None else 0,
        data=serialized_data,
        done=action.done if action.done is not None else False,
        reward_update_flag=action.reward_update_flag if action.reward_update_flag is not None else False,
    )
    return action_proto

def deserialize_action(action):
    """Deserialize a Protobuf message into an RL4SysAction object."""
    # Handle action deserialization based on the content
    action_bytes = action.action
    if action_bytes == b"None":
        action_value = None
    else:
        try:
            # Try to deserialize as numpy array first
            action_value = np.frombuffer(action_bytes)
            if len(action_value) == 1:  # If it's a single value
                action_value = action_value[0]
        except:
            # If that fails, try as integer
            try:
                action_value = int.from_bytes(action_bytes, byteorder='big', signed=True)
            except:
                print(f"Warning: Could not deserialize action bytes: {action_bytes}")
                action_value = None

    # Deserialize data dictionary values from bytes
    deserialized_data = {}
    for k, v in action.data.items():
        try:
            # Try to deserialize as numpy array
            arr = np.frombuffer(v, dtype=np.float32)
            if len(arr) == 1:
                deserialized_data[k] = float(arr[0])
            else:
                deserialized_data[k] = arr
        except:
            # If that fails, try as string
            try:
                deserialized_data[k] = v.decode('utf-8')
            except:
                print(f"Warning: Could not deserialize data value for key {k}")
                deserialized_data[k] = None

    return RL4SysAction(
        obs=deserialize_tensor(action.obs),
        action=action_value,
        mask=deserialize_tensor(action.mask),
        reward=action.reward,
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


