from _common._algorithms.BaseKernel import mlp, ForwardKernelAbstract, StepKernelAbstract

from typing import Optional, Type

import torch
import torch.nn as nn
from numpy import ndarray


class PolicyNetwork(ForwardKernelAbstract):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.pi_network = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, act_dim),
        )

    def _distribution(self, obs: torch.Tensor, mask: torch.Tensor):
        x = obs.view(-1, obs.size(-1), obs.shape[1])
        x = self.pi_network(x)
        logits = torch.squeeze(x, -1)
        logits = logits + (mask - 1) * 1e8
        return logits

    def _log_prob_from_distribution(self, pi, act):
        log_probs = torch.log_softmax(pi, dim=-1)
        return log_probs.gather(-1, act)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, act: Optional[torch.Tensor] = None):
        pi = self._distribution(obs, mask)

        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
            return logp_a

        return pi, None


class BaselineValueNetwork(ForwardKernelAbstract):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_network = mlp([obs_dim] + hidden_sizes + [1], activation)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        return torch.squeeze(self.v_network(obs), -1)


class PolicyWithBaseline(StepKernelAbstract):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.policy = PolicyNetwork(obs_dim, act_dim)
        self.baseline = BaselineValueNetwork(obs_dim, hidden_sizes, activation)

    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        logp_a = self.policy(obs, mask)
        return logp_a, {}


class PolicyWithoutBaseline(StepKernelAbstract):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.policy = PolicyNetwork(obs_dim, act_dim)

    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        logp_a = self.policy(obs, mask)
        return logp_a, {}