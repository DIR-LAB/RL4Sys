from typing import Optional, Type

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from algorithms._common.BaseKernel import mlp, ForwardKernelAbstract, StepKernelAbstract

"""
Network configurations for SAC
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class RLActor(ForwardKernelAbstract):
    """
    Squashed gaussian MLP actor network.
    """
    def __init__(self, obs_dim: int, act_dim: int, act_limit: int, hidden_sizes: tuple = (256, 256),
                 activation=nn.ReLU):
        super().__init__()
        self.obs_dim = obs_dim
        self.layer_sizes = [obs_dim] + list(hidden_sizes)

        self.activation = activation
        self.actor_net = mlp([obs_dim] + list(hidden_sizes), self.activation)

        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, deterministic: Optional[bool] = False,
                with_logprob: Optional[bool] = True):
        act = self.actor_net(obs)
        mu = self.mu_layer(act)

        log_std = self.log_std_layer(act)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_a = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_a -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1)
        else:
            logp_a = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_a


class QFunction(ForwardKernelAbstract):
    """
    MLP Q-network.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)


class RLActorCritic(StepKernelAbstract):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        act_limit = act_dim

        self.pi = RLActor(obs_dim, act_dim, act_limit, hidden_sizes, activation)
        self.q1 = QFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = QFunction(obs_dim, act_dim, hidden_sizes, activation)

    def step(self, obs: torch.Tensor, mask: torch.Tensor, deterministic: Optional[bool] = False):
        with torch.no_grad():
            a, logp_a = self.pi.forward(obs, mask, deterministic, True)
            data = {'logp_a': logp_a}
            return a.numpy(), data
