from abc import ABC, abstractmethod
from typing import Union


class RL4SysTrainingServerAbstract(ABC):
    def __init__(self, algorithm_name: str, obs_size: int, obs_dim: int, hyperparams: Union[dict | list[str]],
                 *args, **kwargs):
        super(RL4SysTrainingServerAbstract, self).__init__()

    @abstractmethod
    def send_model(self, *args, **kwargs):
        """

        """
        pass

    @abstractmethod
    def start_loop(self, *args, **kwargs):
        """Used for listening for data meant for this server.

        Typically used for listening for trajectories.
        Primary source of server activity
        """
        pass
