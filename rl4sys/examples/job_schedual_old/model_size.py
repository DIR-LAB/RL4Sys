import torch
import torch.nn as nn
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from rl4sys.algorithms.PPO.kernel import RLActorCritic

def human(n):
    for u in ['B','KiB','MiB','GiB','TiB']:
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PiB"

input_size = 1024
act_dim = 128
batch = 25600
dtype = torch.float32
device = 'cuda'  # change to 'cuda' if available

model = RLActorCritic(input_size, act_dim, actor_type="kernel").to(device, dtype=dtype)

# dummy inputs
obs = torch.randn(batch, input_size, dtype=dtype, device=device)
mask = torch.ones(batch, act_dim, dtype=torch.bool, device=device)  # or 0/1 mask

with torch.no_grad():
    actions, info = model.step(obs, mask=mask.to(dtype=torch.float32))  # if your mask expects float
print("Forward done.")

print("Memory allocated to CUDA:", torch.cuda.memory_allocated() / 1024 / 1024, "MB")