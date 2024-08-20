from _common._algorithms.BaseKernel import StepAndForwardKernelAbstract, mlp

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

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
    def __init__(self, kernel_dim: int, kernel_size: int, act_dim: int, atoms, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 5e-4,
                 custom_network: nn.Sequential = None):
        super().__init__()
        if custom_network is None:
            self.q_network = mlp([kernel_dim * kernel_size] + [32, 16, 8] + [act_dim * len(atoms)], nn.ReLU)
        else:
            self.q_network = custom_network

        self.kernel_dim = kernel_dim
        self.kernel_size = kernel_size
        self.act_dim = act_dim

        self.atoms = atoms

        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

    def _distribution(self, obs: torch.Tensor, mask: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """
            Forward pass through Q-network, outputs Q-values for actions.
        Args:
            obs: current observation
            mask: mask for current observation (unused in C51)
        Returns:
            Q-values for actions
        """

        logits = self.q_network(obs)
        q_pmf = Categorical(logits=logits)
        q_vals = (q_pmf.logits * self.atoms)

        return q_pmf, q_vals

    def _log_prob_from_distribution(self, pi: torch.distributions.distribution.Distribution, act) -> torch.Tensor:
        """
        Get log probability of action(s) from distribution
        Args:
            pi: distribution
            act: action(s)
        Returns:
            log probability of action(s)
        """
        if not isinstance(act, torch.Tensor):
            act = torch.tensor(act)
        return pi.log_prob(act)

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

        q_pmf, q_vals = self._distribution(obs, mask)

        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(q_pmf, act)

        return q_pmf, q_vals, logp_a

    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        """
            Select an action based on epsilon-greedy strategy.
            if explore, select random action, else select action with highest Q-value.
        Args:
            obs: current observation
            mask: mask for current observation (unused in C51)
        Returns:

        """

        with torch.no_grad():
            q_pmf, q_vals, _ = self.forward(obs, mask)
            if np.random.rand() <= self._epsilon:
                a = np.random.randint(0, self.kernel_size)
            else:
                a = torch.argmax(q_vals).item()
            logp_a = self._log_prob_from_distribution(q_pmf, a)
        data = {'q_vals': q_vals.detach().numpy(), 'logp_a': logp_a.detach().numpy(), 'epsilon': self._epsilon}
        self._epsilon = max(self._epsilon - self._epsilon_decay, self._epsilon_min)

        return a, data
