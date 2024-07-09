from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as functional


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
    def __init__(self, obs_dim: int, obs_size: int, act_dim: int = 1):
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim * obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, act_dim)
        )

        self.obs_dim = obs_dim
        self.obs_size = obs_size
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None, softmax: bool = False):
        """

        Args:
            obs:
            mask:
            softmax:
        Returns:

        """
        # if mask is not None:
        #     obs = obs * mask

        q = self.q_network(obs)
        if softmax:
            return functional.softmax(q, dim=0)
        else:
            return q

    def step(self, obs: torch.Tensor, mask: torch.Tensor = None):
        """

        Args:
            obs:
            mask:
        Returns:

        """
        with torch.no_grad():
            q = self.forward(obs, mask, softmax=True)
            a = torch.argmax(q)
        data = {'q_val': q.numpy()}
        return a, data
