from abc import ABC, abstractmethod

from _common._rl4sys.BaseTrajectory import RL4SysTrajectoryAbstract


class AlgorithmAbstract(ABC):
    def __init__(self):
        super(AlgorithmAbstract, self).__init__()

    @abstractmethod
    def save(self, filename) -> None:
        pass

    @abstractmethod
    def receive_trajectory(self, trajectory: RL4SysTrajectoryAbstract) -> bool:
        pass

    @abstractmethod
    def train_model(self) -> None:
        pass

    @abstractmethod
    def log_epoch(self) -> None:
        pass

