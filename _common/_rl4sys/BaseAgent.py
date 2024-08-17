from abc import ABC, abstractmethod

import torch


class RL4SysAgentAbstract(ABC):
    """
    Abstract class for an agent in RL4Sys.
    """
    def __init__(self, model: torch.nn.Module, training_server_port: int, *args, **kwargs):
        super(RL4SysAgentAbstract, self).__init__()

    @abstractmethod
    def request_for_action(self, obs: torch.tensor, mask: torch.tensor, reward, *args, **kwargs):
        """
        Request an action from the agent's model.
        """
        pass

    @abstractmethod
    def flag_last_action(self, reward: int):
        """
        Flag the last action as done and with a reward.
        """
        pass

    @abstractmethod
    def _loop_for_updated_model(self):
        """
        Loop/listen for updated model from training server. Or else where... ?
        """
        pass
