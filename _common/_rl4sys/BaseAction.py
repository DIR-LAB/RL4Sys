from abc import ABC, abstractmethod


class ActionAbstract(ABC):
    def __init__(self, *args, **kwargs):
        super(ActionAbstract, self).__init__()

    @abstractmethod
    def update_reward(self, *args, **kwargs):
        pass
