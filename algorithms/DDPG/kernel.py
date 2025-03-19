from _common._algorithms.BaseKernel import StepAndForwardKernelAbstract
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(StepAndForwardKernelAbstract):
    def __init__(self, input_size: int, act_dim: int, act_limit: float):
        super().__init__()
        # Match cleanRL architecture
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, act_dim)
        
        # Action scaling
        self.action_scale = act_limit
        self.action_bias = 0.0

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

    def step(self, obs: torch.Tensor, mask: torch.Tensor = None):
        with torch.no_grad():
            action = self.forward(obs, mask)
            # Add exploration noise like cleanRL
            action += torch.normal(0, self.action_scale * self.noise_scale)
            action = action.clamp(-self.action_scale, self.action_scale)
        return action.numpy(), {}

class Critic(nn.Module):
    def __init__(self, input_size: int, act_dim: int):
        super().__init__()
        # Match cleanRL architecture
        self.fc1 = nn.Linear(input_size + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)
