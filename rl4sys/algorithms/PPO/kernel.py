from _common._algorithms.BaseKernel import mlp, ForwardKernelAbstract, StepKernelAbstract

from typing import Optional, Type, Tuple

import torch
import torch.nn as nn
from numpy import ndarray
from torch.distributions.categorical import Categorical
import numpy as np

"""
Network configurations for PPO
"""


class RLActorCritic(nn.Module):
    def __init__(self, input_size, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, mask=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a linear layer using orthogonal initialization, then set bias.
    By default, std is sqrt(2) (common in orthogonal init for ReLU/Tanh).
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
