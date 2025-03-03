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


class RLActor(ForwardKernelAbstract):
    """Neural network of Actor.

    Produces distributions for actions.

    Attributes:
        kernel_size (int): Number of observations. e.g. MAX_QUEUE_SIZE
        kernel_dim (int): Number of features. e.g. JOB_FEATURES

    Neural network:
        input size: kernel_dim
        hidden layer sizes: (32, 16, 8)
        output size: 1
        activation layer: ReLU
        output activation layer: none

    Neural network input:
        Flattened tensor, with original shape (kernel_size, kernel_dim)
        Dimension 0 represents actions in observation
        Dimension 1 represent action features

        Ex. (should be flattened before passing as input)
        [[action1.feature1, action1.feature2],
         [action2.feature1, action2.feature2],]

    Neural network output:
        Tensor of shape (kernel_size, 1), which should then be squeezed on dim=-1 to size (kernel_size)
        Represents logits distribution

        Ex.
        [ [action1_logit],
          [action2_logit] ]

    """

    def __init__(self, input_size: int, act_dim: int, custom_network: nn.Sequential = None):
        super().__init__()
        if custom_network is None:
            self.pi_network = nn.Sequential(
                layer_init(nn.Linear(input_size, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, act_dim), std=0.01)
            )
        else:
            self.pi_network = custom_network

        self.input_size = input_size
        self.act_dim = act_dim

    def _distribution(self, flattened_obs: torch.Tensor, mask: torch.Tensor) -> Categorical:
        """Get actor policy for a given observation.
        
        Builds an new MLP using neural network modules declared in object instance, and ReLU activation layers.

        Args:
            flattened_obs: a tensor of shape (kernel_size, kernel_dim) that has been flattened
            mask: a tensor of shape (kernel_size)
        Returns:
            torch Categorical distribution correpsonding to action probabilities

        """

        #print('whats flattened obs? ', flattened_obs)
        # x = flattened_obs.view(-1, self.input_size) # unclear reason for -1 dimension
        x = self.pi_network(flattened_obs)
        #print('what is x now:',logits)

        return Categorical(logits=x)

    def _log_prob_from_distribution(self, pi: torch.distributions.distribution.Distribution, act: torch.Tensor) -> torch.Tensor:
        """Get log of the probability for specific action in a distribution.

        Can be useful in computing loss function.

        Args:
            pi: distribution
            act: action(s), as corresponding index values in distribution
        Returns:
            log_prob for action(s)

        """
        return pi.log_prob(act)

    def forward(self, flattened_obs: torch.Tensor, mask: torch.Tensor, act: Optional[torch.Tensor] = None) -> tuple[Categorical, Optional[torch.Tensor]]:
        """Get agent policy

        Can return log probabilities for actions if desired, by passing parameter for act.

        Args:
            flattened_obs: a tensor of shape (kernel_size, kernel_dim) that has been flattened
            mask: a tensor of shape (kernel_size)
            act: action(s), as index values in unflattened obs tensor, for which to take log probability
        Returns:
            Policy as a Categorical distribution, as well as log_pi for action(s)

        """

        pi = self._distribution(flattened_obs, mask)

        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        
        return pi, logp_a

    def step(self, flattened_obs: torch.Tensor, mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        A simple step function for inference only, returning an action and
        its log-prob under the current actor network.

        Args:
            flattened_obs (torch.Tensor): Flattened observation of shape (kernel_size, kernel_dim).
            mask (torch.Tensor): A mask of shape (kernel_size) indicating valid vs. invalid actions.

        Returns:
            A tuple containing:
                - action (np.ndarray): Sampled action index.
                - logp (np.ndarray): Log prob of the sampled action.
        """
        with torch.no_grad():
            pi, _ = self.forward(flattened_obs, mask)
            action = pi.sample()                         # sample an action
            logp = self._log_prob_from_distribution(pi, action)  # log prob of that action

        data = {'logp_a': logp.numpy()}

        return action.numpy(), data


class RLCritic(ForwardKernelAbstract):
    """Neural network of Critic.

    Produces an estimate for V (state-value).

    Attributes:
        obs_dim (int): length of the observation. NOTE: observation should be flattened.

    Neural network:
        input size: obs_dim
        hidden layer sizes: default (32, 16, 8)
        output size: 1
        activation layer: default ReLU
        output activation layer: nn.Identity

    Neural network input:
        Flattened tensor, should be shape (kernel_size * kernel_dim)

        Ex.
        [action1.feature1, action1.feature2, action2.feature1, action2.feature2]

    Neural network output:
        Tensor of rank 0 (scalar).

    """

    def __init__(self, obs_dim: int, custom_network: nn.Sequential = None):
        super().__init__()
        self.obs_dim = obs_dim
        if custom_network is None:
            self.v_net = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)), 
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0)
            )
        else:
            self.v_net = custom_network

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Get estimate for state-value

        mask is not relevant in critic model but retained for consistency.

        Args:
            obs: a tensor of shape (kernel_size * kernel_dim)
            mask: unused
        Returns:
            Tensor of rank 0 (scalar).

        """
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)

    def step(self, obs: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
        """
        A simple step function for inference only, returning the critic's value estimate.

        Args:
            obs (torch.Tensor): Flattened observation of shape (kernel_size * kernel_dim).
            mask (torch.Tensor): A mask of shape (kernel_size), not used here but kept for consistency.

        Returns:
            A numpy array (with shape (kernel_size,) if batching) representing
            the critic's value estimate (V) for each observation in the batch.
        """
        with torch.no_grad():
            v = self.forward(obs, mask)
        return v.numpy()


class RLActorCritic(nn.Module):
    """
    A single PPO Actor-Critic network, similar to CleanRL's approach.

    Attributes:
        actor (nn.Sequential):   Produces action logits.
        critic (nn.Sequential):  Produces state-value estimates.

    Example usage:
        agent = RLActorCritic(input_size=8, act_dim=4)
        # Forward pass:
        obs = torch.randn((4, 8))   # batch of 4 with obs_dim=8
        action, log_prob, entropy, value = agent.get_action_and_value(obs)
    """
    def __init__(self, input_size: int, act_dim: int):
        super().__init__()
        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the value estimate (V) for the environment's state.

        Args:
            x (torch.Tensor): Network input of shape [batch_size, input_size].
        Returns:
            A tensor of shape [batch_size] with the value estimates.
        """
        return self.critic(x).squeeze(-1)

    def step(
        self, 
        x: torch.Tensor, 
        action: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute action, log probability of that action, the policy's entropy,
        and the value estimate, all in one forward pass.

        Args:
            x (torch.Tensor): [batch_size, input_size] observation tensor.
            action (Optional[torch.Tensor]): If given, we compute log_prob
                for that action instead of sampling.

        Returns:
            action (torch.Tensor): Sampled or provided discrete action.
            logprob (torch.Tensor): The log probability of that action.
            entropy (torch.Tensor): The policy entropy for exploration measure.
            value (torch.Tensor): Value estimate, shape [batch_size].

        Example:
            act, logp, ent, val = agent.get_action_and_value(obs)
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a linear layer using orthogonal initialization, then set bias.
    By default, std is sqrt(2) (common in orthogonal init for ReLU/Tanh).
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
