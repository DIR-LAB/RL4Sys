from abc import ABC, abstractmethod
import torch.nn as nn
import torch


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


def mlp(sizes, activation, output_activation=nn.Identity):
    """Build a multilayer perceptron, with layers of nn.Linear.

    Args:
        sizes: a tuple of ints, each declaring the size of one layer.
        activation: function type for activation layer.
    Returns:
        the built neural network as a torch.nn.Module.

    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class ForwardKernelAbstract(nn.Module, ABC):
    """
    PyTorch NN module class for forward pass.
    """
    def __init__(self, custom_network: nn.Sequential = None):
        super(ForwardKernelAbstract, self).__init__()

    @abstractmethod
    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        """
        Pass obs and mask through the forward pass of kernel's network.
        """
        pass


class StepKernelAbstract(nn.Module, ABC):
    """
    PyTorch NN module class for step function.
    """
    def __init__(self, custom_network: nn.Sequential = None):
        super(StepKernelAbstract, self).__init__()
        self.custom_network = custom_network

    @abstractmethod
    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        """
        Pass obs and mask through the step function of kernel's network. May have different behavior
        than forward function.
        """
        pass


class StepAndForwardKernelAbstract(nn.Module, ABC):
    """
    PyTorch NN module class for step and forward functions.
    """
    def __init__(self, custom_network: nn.Sequential = None):
        super(StepAndForwardKernelAbstract, self).__init__()

    @abstractmethod
    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        """
        Pass obs and mask through the forward pass of kernel's network.
        """
        pass

    @abstractmethod
    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        """
        Pass obs and mask through the step function of kernel's network. May have different behavior
        than forward function.
        """
        pass
