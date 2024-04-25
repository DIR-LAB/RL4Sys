from typing import Optional

class RL4SysAction:
    """An action stored with relevant data for RL models.

    Attributes:
        obs (torch.Tensor): flattened observation, used as input for kernels. Before flattening, should have shape (KERNEL_SIZE, KERNEL_DIM).
        action (torch.Tensor): the action chosen by RL4SysAgent. Should be a rank-0 Tensor (scalar), repesenting action's index in self.obs when unflattened.
        mask (torch.Tensor): observation mask. Should have shape (KERNEL_SIZE). See RL4SysAgent for details.
        reward (int): reward for taking the action which led to this observation. NOTE: this is not the reward for taking the current action, but in fact the previous in the trajectory.
        data (dict): extra values from model.step() which the selected algorithm requires, such as log probability or state value.
        done (bool): True for last action in trajectory, in which case all fields should be None except for reward.

    """
    # TODO find out what reward's actual type should be. numpy float64? update type hints throughout project
    # TODO shorten docstrings like this one to fit within required 72 characters
    # TODO replace all forward evaluation type hints like these by imports that are only called by type checkers
    def __init__(self, obs: Optional['torch.Tensor'], action: Optional['torch.Tensor'], mask: Optional['torch.Tensor'], reward: int, data: Optional[dict], done: bool):
        self.obs = obs
        self.act = action
        self.mask = mask
        self.rew = reward
        self.data = data
        self.done = done

    def update_reward(self, reward: int) -> None:
        self.rew = reward