from abc import ABC, abstractmethod


class RL4SysActionAbstract(ABC):
    def __init__(self, *args, **kwargs):
        super(RL4SysActionAbstract, self).__init__()

    @abstractmethod
    def update_reward(self, *args, **kwargs):
        pass
