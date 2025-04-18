from _common._rl4sys.BaseTrajectory import RL4SysTrajectoryAbstract

from abc import ABC, abstractmethod


class AlgorithmAbstract(ABC):
    """
    Abstract class for designing and implementing new RL algorithms that can interface
    with RL4Sys.
    """
    def __init__(self):
        super(AlgorithmAbstract, self).__init__()
        
        


    @abstractmethod
    def save(self, filename) -> None:
        """
        Save the model to a file of a specific filename.
        """
        pass

    @abstractmethod
    def receive_trajectory(self, trajectory: RL4SysTrajectoryAbstract) -> bool:
        """
        Receive a trajectory from the environment and store it in the replay buffer.
        """
        pass

    @abstractmethod
    def train_model(self) -> None:
        """
        Train the model on the retrieved, stored trajectories.
        """
        pass

    @abstractmethod
    def log_epoch(self) -> None:
        """
        Log the current epoch and its statistics.
        """
        pass

