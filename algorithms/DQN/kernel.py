from typing import Optional, Type

import torch
import torch.nn as nn


"""
Network configurations for DQN
"""


class DeepQNetwork(nn.Module):
    """Neural network for DQN.

    Produces Q-values for actions; return categorical distribution if cat=True

        Args:
            obs_dim:
            obs_size:
    """
    def __init__(self, obs_dim: int, obs_size: int, act_dim: int = 1, cat: bool = False):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, act_dim)
        )

        self.obs_dim = obs_dim
        self.obs_size = obs_size
        self.act_dim = act_dim
        self.cat = cat

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        """

        Args:
            obs:
            mask:

        Returns:

        """

        # if mask is not None:
        #     return self.q_network(obs, mask)

        return self.q_network(obs)

    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        """

        Args:
            obs:
            mask:
        Returns:

        """
        with torch.no_grad():
            q = self.forward(obs)
            a = torch.argmax(q).item()
        a = a.numpy()
        data = {'q_val': q.numpy()}
        return a, data
