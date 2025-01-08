from _common._algorithms.BaseKernel import ForwardKernelAbstract, StepKernelAbstract, mlp

from typing import Optional, Type

import torch
import torch.nn as nn
from numpy import ndarray


class Actor(ForwardKernelAbstract):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes, activation, act_limit):
        super().__init__()
        self.model = mlp([obs_dim] + list(hidden_sizes), activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, act: torch.Tensor = None):
        return self.act_limit * self.model(obs)


class QCritic(ForwardKernelAbstract):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.model = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, act: torch.Tensor = None):
        q = self.model(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class ActorCritic(StepKernelAbstract):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q_critic1 = QCritic(obs_dim, act_dim, hidden_sizes, activation)
        self.q_critic2 = QCritic(obs_dim, act_dim, hidden_sizes, activation)

    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        with torch.no_grad():
            act = self.actor.model(obs)
        return act.numpy(), {}
