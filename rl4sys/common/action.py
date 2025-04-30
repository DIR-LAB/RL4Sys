from typing import Optional

import torch

class RL4SysAction():
    """
    An action stored with relevant data for RL models.

    Attributes:
        obs (torch.Tensor): flattened observation, used as input for kernels.
            Before flattening, should have shape (KERNEL_SIZE, KERNEL_DIM).
        action (torch.Tensor): the action chosen by RL4SysAgent. Should be a 
            rank-0 Tensor (scalar), representing action's index in self.obs 
            when unflattened.
        reward (bytes): reward for taking the action which led to this observation.
            NOTE: this is not the reward for taking the current action, but in 
            fact the previous in the trajectory.
        done (bool): True for last action in trajectory, in which case all fields 
            should be None except for reward.
        data (dict): extra values from model.step() which the selected algorithm 
            requires, such as log probability or state value.
        version (int): version of the model that generated this action
    """
    def __init__(self, 
                 obs: Optional['torch.Tensor'] = None, 
                 action: Optional['torch.Tensor'] = None, 
                 reward: Optional['torch.Tensor'] = None, 
                 done: bool = None,
                 data: Optional[dict] = {},
                 version: int = 0):
        super().__init__()
        self.obs = obs
        self.act = action
        self.rew = reward
        self.done = done
        self.data = data
        self.version = version

    def update_reward(self, reward: Optional['torch.Tensor']) -> None:
        self.rew = reward

    def set_done(self, done: bool) -> None:
        self.done = done

    def is_reward_set(self) -> bool:
        return self.rew is not None

    def is_done(self) -> bool:
        return self.done
