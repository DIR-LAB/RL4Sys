from abc import ABC, abstractmethod
from typing import Union


class TrainingServerAbstract(ABC):
    def __init__(self, algorithm_name: str, obs_size: int, obs_dim: int, hyperparams: Union[dict | list[str]],
                 *args, **kwargs):
        super(TrainingServerAbstract, self).__init__(algorithm_name, obs_size, obs_dim, hyperparams, *args, **kwargs)

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
