from _common._algorithms.BaseKernel import mlp, ForwardKernelAbstract, StepKernelAbstract

from typing import Optional, Type

import torch
import torch.nn as nn
from numpy import ndarray
from torch.distributions.categorical import Categorical

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

    def __init__(self, input_size: int, act_dim: int , custom_network: nn.Sequential = None):
        super().__init__()
        if custom_network is None:
            self.pi_network = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, act_dim)
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

    def __init__(self, obs_dim: int, hidden_sizes: tuple[int, int, int] = (32, 16, 8), activation: Type[nn.Module] = nn.ReLU,
                 custom_network: nn.Sequential = None):
        super().__init__()
        self.obs_dim = obs_dim
        if custom_network is None:
            self.v_net = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
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


class RLActorCritic(StepKernelAbstract):
    """PPO Actor-Critic kernel.

    Attributes:
        flatten_obs_dim (int): length of the observation when flattened
        kernel_size (int): Number of actions in observation. e.g. MAX_QUEUE_SIZE
        kernel_dim (int): Number of features. e.g. JOB_FEATURES

        pi (RLActor): actor neural net
        pi (RLActor): critic neural net

    """

    def __init__(self, input_size: int, act_dim: int):
        super().__init__()
        self.flatten_obs_dim = input_size
        self.act_dim = act_dim

        # build actor function
        self.pi = RLActor(self.flatten_obs_dim, act_dim)
        # build value function
        self.v = RLCritic(self.flatten_obs_dim)

    def step(self, flattened_obs: torch.Tensor, mask: torch.Tensor) -> (tuple[ndarray, dict[str, ndarray]] |
                                                                        tuple[Type[ndarray], Type[dict[str, ndarray]]]):
        """Get estimate for state-value

        Mask should contain 1 for all actions which are able to be chosen, and 0 for disabled.
        For example, if kernel_size is 6 but only 4 actions are available in this observation, mask unused spots:
            [1, 1, 1, 1, 0, 0]

        Args:
            flattened_obs: a tensor of shape (kernel_size, kernel_dim) that has been flattened
            mask: a tensor of shape (kernel_size)
        Returns:
            policy-chosen action
            dict:
                {'v'      : state-value V,
                 'logp_a' : log probability of chosen action}

        """
        # TODO masks may actually be ndarray type?
        # TODO a might just return an index. make sure that's ok
        if flattened_obs is not None and mask is not None:
            with torch.no_grad():
                # actor
                pi, _ = self.pi(flattened_obs, mask)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                # critic
                v = self.v(flattened_obs, mask)
            a = a.numpy()
            data = {'v': v.numpy(), 'logp_a': logp_a.numpy()}
            return a, data
        return ndarray, dict[str, ndarray]

    # this method appears completely unused by the training server code
    def act(self, flattened_obs: torch.Tensor, mask: torch.Tensor) -> ndarray:
        """Select an action according to the learned policy.

        Args:
            flattened_obs: a tensor of shape (kernel_size, kernel_dim) that has been flattened
            mask: a tensor of shape (kernel_size)
        Returns:
            Action as numpy array

        """

        with torch.no_grad():
            distribution, _ = self.pi(flattened_obs, mask)[0]
            action = distribution.sample()
            return action.numpy()
