from _common._algorithms.BaseKernel import StepAndForwardKernelAbstract, mlp

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

"""
Network configurations for C51
"""


class CategoricalQNetwork(StepAndForwardKernelAbstract):
    """Neural network for C51.

    Produces softmax distribution over Q-values for actions.
    Uses epsilon-greedy strategy for action exploration-exploitation process.

        Args:
            kernel_dim: number of observations
            kernel_size: number of features
            act_dim: number of actions (output layer dimensions)
            epsilon: Initial value for epsilon; exploration rate that is decayed over time.
            epsilon_min: Minimum possible value for epsilon
            epsilon_decay: Decay rate for epsilon
    """
    def __init__(self, kernel_dim: int, kernel_size: int, act_dim: int, n_atoms, v_min, v_max, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 5e-4,
                 custom_network: nn.Sequential = None):
        super().__init__()
        if custom_network is None:
            self.q_network = mlp([kernel_dim * kernel_size] + [256, 128] + [act_dim * n_atoms], nn.ReLU)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.q_network = custom_network

        self.kernel_dim = kernel_dim
        self.kernel_size = kernel_size
        self.act_dim = act_dim

        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))

        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

    def _distribution(self, obs: torch.Tensor, mask: torch.Tensor):
        """
            Forward pass through Q-network, outputs Q-values for actions.
        Args:
            obs: current observation
            mask: mask for current observation (unused in C51)
        Returns:
            Q-values for actions
        """

        logits = self.q_network(obs)
        logits = logits.view(-1, self.act_dim, self.n_atoms)
        q_pmf = self.softmax(logits)
        q_vals = torch.sum(q_pmf * self.atoms, dim=2)

        return q_pmf, q_vals

    def _log_prob_from_distribution(self, pmf, act) -> torch.Tensor:
        """
        Get log probability of action(s) from distribution
        Args:
            pmf: distribution
            act: action(s)
        Returns:
            log probability of action(s)
        """
        if not isinstance(act, torch.Tensor):
            act = torch.tensor(act)

        return torch.log(act + 1e-8)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None, act: Optional[torch.Tensor] = None):
        """
            Forward pass through Q-network, outputs Q-values for actions.
        Args:
            obs: current observation
            mask: mask for current observation (unused in C51)
            act: action(s) to get log probability for (optional)
        Returns:
            q_pmf: distribution over Q-values
            q_vals: Q-values for actions
            logp_a: log probability of action(s) (optional)
        """

        q_pmf, q_vals = self._distribution(obs, mask)

        if act is None:
            act = torch.argmax(q_vals).item()

        logp_a = self._log_prob_from_distribution(q_pmf, act)

        return q_pmf, q_vals, logp_a

    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        """
            Select an action based on exponential decay epsilon strategy.
            if explore, select random action, else select action with highest Q-value.
        Args:
            obs: current observation
            mask: mask for current observation (unused in C51)
        Returns:

        """

        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        q_pmf, q_vals, _ = self.forward(obs, mask)
        with torch.no_grad():
            if np.random.rand() <= self._epsilon:
                a = np.random.choice(self.kernel_size)
            else:
                a = torch.argmax(q_vals, dim=-1).item()
            logp_a = self._log_prob_from_distribution(q_pmf, a)
        data = {'q_vals': q_vals.detach().numpy(), 'logp_a': logp_a.detach().numpy(), 'epsilon': self._epsilon}

        return a, data
