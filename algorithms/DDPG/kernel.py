import numpy as np

from _common._algorithms.BaseKernel import ForwardKernelAbstract, StepKernelAbstract, mlp

from typing import Optional, Type

import torch
import torch.nn as nn
from numpy import ndarray


class Actor(ForwardKernelAbstract):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes, activation, act_limit):
        super().__init__()
        self.model = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, act: torch.Tensor = None):
        return self.act_limit * self.model(obs)


class QCritic(ForwardKernelAbstract):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.model = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, act: torch.Tensor = None):
        if act is None:
            return None

        obs = torch.as_tensor(obs.clone().detach(), dtype=torch.float32)
        act = torch.as_tensor(act.clone().detach(), dtype=torch.float32)
        q = self.model(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class ActorCritic(StepKernelAbstract):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, act_noise_std):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q_critic = QCritic(obs_dim, act_dim, hidden_sizes, activation)

        self.act_dim = act_dim
        self.act_limit = act_limit
        self.act_noise_std = act_noise_std

    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        with torch.no_grad():
            act = self.actor.forward(obs, mask)
            act += self.act_noise_std * np.random.randn(self.act_dim)
            clipped_act = torch.clip(act, -self.act_limit, self.act_limit)
            q = self.q_critic.forward(obs, mask, clipped_act)
        data = {'q_val': q.detach().numpy(), 'act': act.detach().numpy()}
        return clipped_act, data
