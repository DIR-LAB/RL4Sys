from _common._algorithms.BaseReplayBuffer import combined_shape, discount_cumsum, statistics_scalar, ReplayBufferAbstract

import numpy as np
import torch

"""
SAC Code
"""


class ReplayBuffer(ReplayBufferAbstract):
    """


    """
    def __init__(self, obs_dim, mask_dim, buf_size, gamma):
        super().__init__()
        self.obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(buf_size), dtype=np.float32)
        self.mask_buf = np.zeros(combined_shape(buf_size, mask_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buf_size, dtype=np.float32)
        self.ret_buf = np.zeros(buf_size, dtype=np.float32)
        self.gamma = gamma
        self.ptr, self.path_start_idx, self.max_size = 0, 0, buf_size
        self.capacity = buf_size

    def store(self, obs, act, mask, rew):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.mask_buf[self.ptr] = mask
        self.rew_buf[self.ptr] = rew
        if self.ptr > 0:
            self.next_obs_buf[self.ptr-1] = obs
        self.ptr = (self.ptr+1) % self.max_size
        self.path_start_idx = min(self.path_start_idx+1, self.max_size)

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off by an epoch ending.
        Looks back in buffer to where the trajectory started, and uses the rewards found there to
        calculate the reward-to-go for each state in the trajectory.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)

        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, batch_size: int):
        assert self.ptr < self.max_size
        assert self.ptr >= batch_size
        batch = np.random.randint(0, self.path_start_idx, size=batch_size)
        self.ptr, self.path_start_idx = 0, 0

        data = dict(obs=self.obs_buf[batch],
                    next_obs=self.next_obs_buf[batch],
                    act=self.act_buf[batch],
                    mask=self.mask_buf[batch],
                    rew=self.rew_buf[batch])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}, batch
