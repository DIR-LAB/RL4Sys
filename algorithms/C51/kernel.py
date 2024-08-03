import torch
import torch.nn as nn

import numpy as np

from algorithms._common.BaseKernel import StepAndForwardKernelAbstract

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

        self.atoms = atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=atoms))

        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        """
            Forward pass through Q-network, outputs Q-values for actions.
        Args:
            obs: current observation
            mask: mask for current observation (unused in DQN)
        Returns:
            Q-values for actions
        """
        return self.q_network(obs)

    def step(self, obs: torch.Tensor, mask: torch.Tensor = None):
        """
            Select an action based on epsilon-greedy policy.
            If explore, choose random action from distribution.
            If exploit, choose action with highest Q-value.
        Args:
            obs: current observation
            mask: mask for current observation (unused in DQN)
        Returns:

        """
        if np.random.rand() <= self._epsilon:
            with torch.no_grad():
                logits = self.forward(obs, mask)
                q_pmf = torch.softmax(logits.view(len(obs), self.act_dim, self.n_atoms), dim=2)
                q_distribution = (q_pmf * self.atoms).sum(2)
            a = np.random.choice(self.act_dim * self.atoms, size=len(obs), p=q_distribution)
        else:
            with torch.no_grad():
                logits = self.forward(obs, mask)
                q_pmf = torch.softmax(logits.view(len(obs), self.act_dim, self.n_atoms), dim=2)
                q_distribution = (q_pmf * self.atoms).sum(2)
                a = torch.argmax(q_distribution, dim=1)

        data = {'q_val': q_pmf.detach().numpy()[torch.arange(len(obs)), a], 'epsilon': self._epsilon}
        self._epsilon = max(self._epsilon - self._epsilon_decay, self._epsilon_min)

        return a, data


def infer_next_obs(act, obs: torch.Tensor, mask: torch.Tensor = None):
    """ Placeholder next observation function
    Computes next observation based on current observation and mask.
    Unused in DQN computations.

    Next_obs calculation is the sum of the current observation
     and the action taken in said observation.

    Args:
        act: action taken
        obs: current observation
        mask: mask for current observation (unused in DQN)
    Returns:
        next observation
    """
    next_obs = obs + torch.tensor(act, dtype=torch.float32)
    return next_obs

