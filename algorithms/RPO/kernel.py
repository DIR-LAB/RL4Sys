from _common._algorithms.BaseKernel import StepAndForwardKernelAbstract
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(StepAndForwardKernelAbstract):
    def __init__(self, input_size: int, act_dim: int, rpo_alpha: float):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        
        # Match cleanRL architecture with layer initialization
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, obs: torch.Tensor, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
        else:
            # RPO modification: add uniform noise
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(obs.device)
            action_mean = action_mean + z
            probs = Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def step(self, obs: torch.Tensor, mask: torch.Tensor = None):
        with torch.no_grad():
            action, log_prob, entropy = self.forward(obs)
        return action.numpy(), {'log_prob': log_prob, 'entropy': entropy}

class Critic(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

    def forward(self, obs: torch.Tensor):
        return self.critic(obs)
