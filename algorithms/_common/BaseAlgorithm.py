from abc import ABC, abstractmethod

from trajectory import RL4SysTrajectory

import numpy as np


def count_vars(module) -> int:
    return sum([np.prod(p.shape) for p in module.parameters()])


class AlgorithmAbstract(ABC):
    def __init__(self):
        super(AlgorithmAbstract, self).__init__()

    @abstractmethod
    def save(self, filename) -> None:
        pass

    @abstractmethod
    def receive_trajectory(self, trajectory: RL4SysTrajectory) -> bool:
        pass

    @abstractmethod
    def train_model(self) -> None:
        pass

    @abstractmethod
    def log_epoch(self) -> None:
        pass

