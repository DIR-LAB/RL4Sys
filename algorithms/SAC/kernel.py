from _common._algorithms.BaseKernel import mlp, ForwardKernelAbstract, StepKernelAbstract

from typing import Optional, Type

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

"""
Network configurations for SAC
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class ContinuousSAC(ForwardKernelAbstract):
    """
    Squashed gaussian MLP actor network.
    """

    def __init__(self, obs_dim: int, act_dim: int, act_limit: int, hidden_sizes: tuple = (256, 256),
                 activation=nn.ReLU, log_std_min: int = LOG_STD_MIN, log_std_max: int = LOG_STD_MAX):
        super().__init__()
        self.actor_net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

        self.obs_dim = obs_dim
        self.activation = activation

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.act_limit = act_limit

    def _distribution(self, obs: torch.Tensor, mask: torch.Tensor = None):
        act = self.actor_net(obs)
        mu = self.mu_layer(act)
        log_std = self.log_std_layer(act)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        return Normal(mu, std), mu

    def _log_prob_from_distribution(self, pi: torch.distributions.distribution.Distribution, act: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(act)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None, deterministic: Optional[bool] = False,
                with_logprob: Optional[bool] = True):
        gaussian_distribution, mu = self._distribution(obs, mask)
        if deterministic:
            pi_action = mu
        else:
            pi_action = gaussian_distribution.sample()

        if with_logprob:
            logp_a = self._log_prob_from_distribution(gaussian_distribution, pi_action).sum(axis=-1)
            logp_a -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
        else:
            logp_a = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_a


class DiscreteSAC(ForwardKernelAbstract):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple = (256, 256), activation=nn.ReLU):
        super().__init__()
        self.actor_net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.logit_layer = nn.Linear(hidden_sizes[-1], act_dim)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def _distribution(self, obs: torch.Tensor, mask: torch.Tensor):
        a = self.actor_net(obs)
        logits = self.logit_layer(a)
        if mask is not None:
            logits = logits + (mask-1) * 1e4
        else:
            logits = logits + 1e4

        return Categorical(logits=logits), logits

    def _log_prob_from_distribution(self, pi: torch.distributions.distribution.Distribution, act: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(act)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None, deterministic: Optional[bool] = False,
                with_logprob: Optional[bool] = True):
        categorical_distribution, logits = self._distribution(obs, mask)
        if deterministic:
            pi_action = torch.argmax(logits, dim=-1)
        else:
            pi_action = categorical_distribution.sample()

        if with_logprob:
            logp_a = self._log_prob_from_distribution(categorical_distribution, pi_action)
        else:
            logp_a = None

        return pi_action, logp_a


class QFunction(ForwardKernelAbstract):
    """
    MLP Q-network.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, discrete=False):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

        self.discrete = discrete

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        act = act.unsqueeze(1)
        obs = torch.cat([obs, act], dim=1)
        q_vals = self.q(obs)
        if not self.discrete:
            q_vals = torch.squeeze(q_vals, -1)

        return q_vals


class RLActorCritic(StepKernelAbstract):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), activation=nn.ReLU, log_std_min: int = LOG_STD_MIN,
                 log_std_max: int = LOG_STD_MAX, discrete=False):
        super().__init__()
        act_limit = act_dim

        if discrete:
            self.pi = DiscreteSAC(obs_dim, act_dim, hidden_sizes, activation)
        else:
            self.pi = ContinuousSAC(obs_dim, act_dim, act_limit, hidden_sizes, activation, log_std_min, log_std_max)
        self.q1 = QFunction(obs_dim, act_dim, hidden_sizes, activation, discrete)
        self.q2 = QFunction(obs_dim, act_dim, hidden_sizes, activation, discrete)

    def step(self, obs: torch.Tensor, mask: torch.Tensor, deterministic: Optional[bool] = False):
        with torch.no_grad():
            a, logp_a = self.pi.forward(obs, mask, deterministic, True)
            data = {'logp_a': logp_a}
            return a.numpy(), data
