from _common._algorithms.BaseKernel import StepAndForwardKernelAbstract
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(StepAndForwardKernelAbstract):
    def __init__(self, input_size: int, act_dim: int, act_limit: float):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, act_dim)
        
        # Action scaling
        self.register_buffer(
            "action_scale",
            torch.tensor(act_limit, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(0.0, dtype=torch.float32)
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

    def step(self, obs: torch.Tensor, mask: torch.Tensor = None):
        with torch.no_grad():
            action = self.forward(obs)
            # Add exploration noise
            action += torch.normal(0, self.action_scale * self.noise_scale)
            action = action.clamp(-self.action_scale, self.action_scale)
        return action.numpy(), {}

class Critic(nn.Module):
    def __init__(self, input_size: int, act_dim: int):
        super().__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(input_size + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        # Q2 architecture
        self.fc4 = nn.Linear(input_size + act_dim, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        
        # Q1
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        # Q2
        q2 = F.relu(self.fc4(x))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2

    def Q1(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1
