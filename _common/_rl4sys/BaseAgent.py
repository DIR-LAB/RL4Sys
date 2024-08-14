from abc import ABC, abstractmethod

import torch


class AgentAbstract(ABC):
    def __init__(self, model: torch.nn.Module, training_server_port: int, tensorboard: bool):
        super(AgentAbstract, self).__init__(model, training_server_port, tensorboard)

    @abstractmethod
    def request_for_action(self, obs: torch.tensor, mask: torch.tensor, reward, *args, **kwargs):
        pass

    @abstractmethod
    def flag_last_action(self, reward: int):
        pass

    @abstractmethod
    def _loop_for_updated_model(self):
        pass
