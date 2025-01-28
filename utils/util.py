import io
import torch
from protocol.action import RL4SysAction


# for gRPC transfering. Torch tensors will be conver into bytes
def serialize_tensor(tensor):
    """Serialize a PyTorch tensor to bytes."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()

def deserialize_tensor(tensor_bytes):
        """Deserialize bytes back to a PyTorch tensor."""
        if tensor_bytes:  # Check if tensor_bytes is not empty
            buffer = io.BytesIO(tensor_bytes)
            return torch.load(buffer, weights_only=True)
        return None

def deserialize_action(action):
    """Deserialize a Protobuf message into an RL4SysAction object."""
    return RL4SysAction(
        obs=deserialize_tensor(action.obs),
        action=deserialize_tensor(action.action),
        mask=deserialize_tensor(action.mask),
        reward=action.reward,
        data=dict(action.data),  # Convert Protobuf map to Python dict
        done=action.done
    )