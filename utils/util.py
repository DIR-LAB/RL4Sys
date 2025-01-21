import io
import torch


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


# TBD
