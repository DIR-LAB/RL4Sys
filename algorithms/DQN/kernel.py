import torch
import torch.nn as nn

import numpy as np


"""
Network configurations for DQN
"""


class DeepQNetwork(nn.Module):
    """Neural network for DQN.

    Produces Q-values for actions; return categorical distribution if cat=True

        Args:
            kernel_dim:
            kernel_size:
    """
    def __init__(self, kernel_dim: int, kernel_size: int, act_dim: int = 1, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 5e-4):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(kernel_dim * kernel_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, act_dim)
        )

        self.kernel_dim = kernel_dim
        self.kernel_size = kernel_size
        self.act_dim = act_dim

        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        """

        Args:
            obs:
            mask:
        Returns:

        """
        # if mask is not None:
        #     obs_float = torch.as_tensor(obs, dtype=torch.float32)
        #     obs_reshaped = obs_float.view(-1, mask.shape[1])
        #     obs = obs_reshaped * mask

        return self.q_network(obs)

    def step(self, obs: torch.Tensor, mask: torch.Tensor = None):
        """

        Args:
            obs:
            mask:
        Returns:

        """
        # if mask is not None:
        #     obs_double = torch.as_tensor(obs, dtype=torch.double)
        #     obs_reshaped = obs_double.view(-1, mask.shape[1])
        #     obs = obs_reshaped * mask

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
        data['next_obs'] = self._compute_next_obs(a, obs, mask)

        return a, data

    def _compute_next_obs(self, act, obs: torch.Tensor, mask: torch.Tensor = None):
        """
        Compute next observation based on current observation and mask.

        Args:
            obs: current observation
            mask: mask for current observation
        Returns:
            next observation
        """
        # if mask is not None:
        #     obs_double = torch.as_tensor(obs, dtype=torch.double)
        #     obs_reshaped = obs_double.view(-1, mask.shape[1])
        #     obs = obs_reshaped * mask

        next_obs = obs + torch.tensor(act, dtype=torch.float32)
        return next_obs

