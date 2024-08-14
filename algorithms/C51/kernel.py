from algorithms._common.BaseKernel import StepAndForwardKernelAbstract

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

import numpy as np

"""
Network configurations for C51
"""


class C51QNetwork(StepAndForwardKernelAbstract):
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
    def __init__(self, kernel_dim: int, kernel_size: int, act_dim: int = 1, atoms: int = 51, v_min: float = -10.0,
                 v_max: float = 10.0, epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 5e-4):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(kernel_dim * kernel_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, act_dim * atoms)
        )

        self.kernel_dim = kernel_dim
        self.kernel_size = kernel_size
        self.act_dim = act_dim

        self.n_atoms = atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=self.n_atoms))

        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

    def _distribution(self, obs: torch.Tensor, mask: torch.Tensor = None) -> Categorical:
        """
            Forward pass through Q-network, outputs Q-values for actions.
        Args:
            obs: current observation
            mask: mask for current observation (unused in C51)
        Returns:
            Q-values for actions
        """
        return Categorical(logits=self.q_network(obs))

    def _log_prob_from_distribution(self, q: torch.distributions.distribution.Distribution, act: torch.Tensor) -> torch.Tensor:
        """
        Get log probability of action(s) from distribution.

        Args:
            q: Q-value distribution
            act: action(s) to get log probability for

        Returns:
            log probability of action(s)
        """
        return q.log_prob(act)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None, act: Optional[torch.Tensor] = None):
        """
            Forward pass through Q-network, outputs Q-values for actions.
        Args:
            obs: current observation
            mask: mask for current observation (unused in C51)
            act: action(s) to get log probability for (optional)
        Returns:
            Q-values for actions
        """
        pmf = self._distribution(obs, mask)

        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pmf, act)

        return pmf, logp_a

    def step(self, obs: torch.Tensor, mask: torch.Tensor = None):
        """
            Select an action based on epsilon-greedy policy.
            If explore, choose random action from distribution.
            If exploit, choose action with highest Q-value from distribution.
        Args:
            obs: current observation
            mask: mask for current observation (unused in C51)
        Returns:

        """
        with torch.no_grad():
            q_pmf = self.forward(obs, mask)[0]
            if np.random.rand() <= self._epsilon:
                a = np.random.choice(q_pmf.logits.size(0))
            else:
                a = torch.argmax(q_pmf.sample(), dim=0).item()
            q_logp_a = self._log_prob_from_distribution(q_pmf, torch.tensor([a]))

        data = {'q_pmf': q_pmf, 'logp_a': q_logp_a.numpy(), 'epsilon': self._epsilon}
        self._epsilon = max(self._epsilon - self._epsilon_decay, self._epsilon_min)

        return a, data
