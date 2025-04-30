import io
import torch
import hashlib
import zlib
import threading
from typing import Dict, Tuple, Optional

class ModelDiffManager:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.model_history: Dict[int, Tuple[torch.nn.Module, bytes]] = {}
        self._algorithm_lock = threading.Lock()
        
    def _compute_model_hash(self, model: torch.nn.Module) -> bytes:
        """Compute a hash of the model's state dict."""
        state_dict = model.state_dict()
        # Convert to bytes for hashing
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return hashlib.sha256(buffer.getvalue()).digest()
    
    def _compress_state_dict(self, state_dict: Dict) -> bytes:
        """Compress the state dict for efficient transfer."""
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return zlib.compress(buffer.getvalue())
    
    def _decompress_state_dict(self, compressed_data: bytes) -> Dict:
        """Decompress the state dict."""
        decompressed = zlib.decompress(compressed_data)
        buffer = io.BytesIO(decompressed)
        return torch.load(buffer)
    
    def _compute_parameter_diff(self, old_state: Dict, new_state: Dict) -> Dict:
        """Compute parameter-wise differences between two state dicts."""
        diff = {}
        for key in new_state:
            if key not in old_state or not torch.equal(old_state[key], new_state[key]):
                diff[key] = new_state[key]
        return diff
    
    def add_model_version(self, version: int, model: torch.nn.Module):
        """Add a new model version to history."""
        with self._algorithm_lock:
            if version in self.model_history:
                return
            
            # Compute hash and compress state dict
            model_hash = self._compute_model_hash(model)
            state_dict = model.state_dict()
            compressed_state = self._compress_state_dict(state_dict)
            
            self.model_history[version] = (model, compressed_state)
            
            # Clean up old versions
            if len(self.model_history) > self.max_history:
                oldest_version = min(self.model_history.keys())
                del self.model_history[oldest_version]
    
    def get_model_diff(self, client_version: int, expected_version: int, latest_version: int) -> Optional[Tuple[bytes, int]]:
        """Get the model difference between client version and current version."""
        with self._algorithm_lock:
            if client_version == expected_version:
                return None
            
            if client_version not in self.model_history or expected_version not in self.model_history:
                # If we don't have history, return full model
                _, compressed_state = self.model_history[latest_version]
                return compressed_state, latest_version
            
            # Get the state dicts
            old_model, _ = self.model_history[client_version]
            new_model, _ = self.model_history[expected_version]
            
            # Compute parameter-wise diff
            old_state = old_model.state_dict()
            new_state = new_model.state_dict()
            diff = self._compute_parameter_diff(old_state, new_state)
            
            # Compress and return the diff
            compressed_diff = self._compress_state_dict(diff)
            return compressed_diff, expected_version 