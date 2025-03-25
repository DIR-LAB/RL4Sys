from _common._algorithms.BaseKernel import StepAndForwardKernelAbstract
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a linear layer using orthogonal initialization, then set bias.
    By default, std is sqrt(2) (common in orthogonal init for ReLU/Tanh).
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DDPGActorCritic(nn.Module):
    def __init__(self, input_size: int, act_dim: int, act_limit: float, noise_scale: float = 0.1):
        super().__init__()
        self.actor = Actor(input_size, act_dim, act_limit, noise_scale)
        self.critic = Critic(input_size, act_dim)
        
    def get_action(self, obs: torch.Tensor, mask: torch.Tensor = None):
        return self.actor.step(obs, mask)
    
    def get_value(self, obs: torch.Tensor, act: torch.Tensor):
        return self.critic(obs, act)

class Actor(StepAndForwardKernelAbstract):
    def __init__(self, input_size: int, act_dim: int, act_limit: float, noise_scale: float = 0.1):
        super().__init__()
        # Match cleanRL architecture with layer initialization
        self.fc1 = layer_init(nn.Linear(input_size, 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc_mu = layer_init(nn.Linear(256, act_dim), std=0.01)  # Lower std for final layer
        self.noise_scale = noise_scale
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
        # Match cleanRL architecture with layer initialization
        self.fc1 = layer_init(nn.Linear(input_size + act_dim, 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc3 = layer_init(nn.Linear(256, 1), std=1.0)  # Higher std for value function

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)
    
