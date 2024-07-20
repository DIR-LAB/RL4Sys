from abc import ABC, abstractmethod
import torch.nn as nn
import torch


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


class ForwardKernel(nn.Module, ABC):
    def __init__(self):
        super(ForwardKernel, self).__init__()

    @abstractmethod
    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        pass


class StepKernel(nn.Module, ABC):
    def __init__(self):
        super(StepKernel, self).__init__()

    @abstractmethod
    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        pass


class StepForwardKernel(nn.Module, ABC):
    def __init__(self):
        super(StepForwardKernel, self).__init__()

    @abstractmethod
    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        pass

    @abstractmethod
    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        pass
