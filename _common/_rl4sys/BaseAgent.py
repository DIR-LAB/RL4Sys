from abc import ABC, abstractmethod

import torch


class RL4SysAgentAbstract(ABC):
    def __init__(self, model: torch.nn.Module, training_server_port: int, *args, **kwargs):
        super(RL4SysAgentAbstract, self).__init__()

    @abstractmethod
    def request_for_action(self, obs: torch.tensor, mask: torch.tensor, reward, *args, **kwargs):
        pass

    @abstractmethod
    def flag_last_action(self, reward: int):
        pass

    @abstractmethod
    def _loop_for_updated_model(self):
        pass
