from abc import ABC, abstractmethod


class RL4SysActionAbstract(ABC):
    """
    Abstract class for an action stored with relevant data for RL agent and models.
    """
    def __init__(self, *args, **kwargs):
        super(RL4SysActionAbstract, self).__init__()

    @abstractmethod
    def update_reward(self, *args, **kwargs):
        """
        Update the reward for the action; set self.reward to the new value.
        """
        pass
