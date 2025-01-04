from _common._algorithms.BaseKernel import StepAndForwardKernelAbstract

import torch
import torch.nn as nn
import numpy as np

"""
Network configurations for DQN
"""


class DeepQNetwork(StepAndForwardKernelAbstract):
    """Neural network for DQN.

    Produces Q-values for actions.
    Uses epsilon-greedy strategy for action exploration-exploitation process.

        Args:
            kernel_dim: number of observations
            kernel_size: number of features
            act_dim: number of actions (output layer dimensions)
            epsilon: Initial value for epsilon; exploration rate that is decayed over time.
            epsilon_min: Minimum possible value for epsilon
            epsilon_decay: Decay rate for epsilon
    """
    def __init__(self, kernel_dim: int, kernel_size: int, act_dim: int, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 5e-4, custom_network: nn.Sequential = None):
        super().__init__()
        if custom_network is None:
            self.q_network = nn.Sequential(
                nn.Linear(kernel_dim * kernel_size, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, act_dim)
            )
        else:
            self.q_network = custom_network

        self.kernel_dim = kernel_dim
        self.kernel_size = kernel_size
        self.act_dim = act_dim

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
        Args:
            obs: current observation
            mask: mask for current observation (unused in DQN)
        Returns:

        """
        if np.random.rand() <= self._epsilon:
            with torch.no_grad():
                q = self.forward(obs, mask)
            a = np.random.choice(self.kernel_size)
        else:
            with torch.no_grad():
                q = self.forward(obs, mask)
                a = q.argmax().item()

        data = {'q_val': q.detach().numpy(), 'epsilon': self._epsilon}
        self._epsilon = max(self._epsilon - self._epsilon_decay, self._epsilon_min)

        return a, data
