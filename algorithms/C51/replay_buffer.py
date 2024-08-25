from _common._algorithms.BaseReplayBuffer import combined_shape, discount_cumsum, ReplayBufferAbstract

import numpy as np
import random
import torch

"""
C51 Code
"""


class ReplayBuffer(ReplayBufferAbstract):
    def __init__(self, obs_dim, mask_dim, buf_size, gamma, epsilon):
        super().__init__()
        self.obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(buf_size), dtype=np.int32)
        self.mask_buf = np.zeros(combined_shape(buf_size, mask_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buf_size, dtype=np.float32)
        self.ret_buf = np.zeros(buf_size, dtype=np.float32)
        self.q_val_buf = np.zeros(buf_size, dtype=np.float32)
        self.gamma, self.epsilon = gamma, epsilon
        self.ptr, self.path_start_idx, self.max_size = 0, 0, buf_size
        self.capacity = buf_size

    def store(self, obs, act, mask, rew, q_val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        Stores this observation as the next observation of the previous transition.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.mask_buf[self.ptr] = mask
        self.rew_buf[self.ptr] = rew
        self.q_val_buf[self.ptr] = q_val
        # most accurate way to retrieve next observation, I imagine.
        if self.ptr > 0:
            self.next_obs_buf[self.ptr - 1] = obs
        self.ptr += 1

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
        """ Sample a batch of data from the replay buffer.

        Args:
            batch_size: the number of samples to draw

        Returns:
            A dictionary containing the following keys:
                obs: the current observation
                next_obs: the next observation
                act: the action
                rew: the reward
                ret: the reward-to-go
        """
        assert self.ptr < self.max_size
        assert self.ptr >= batch_size
        # random sample of indices

        batch = np.random.permutation(self.ptr)[:batch_size]
        self.ptr, self.path_start_idx = 0, 0

        data = dict(obs=self.obs_buf[batch], next_obs=self.next_obs_buf[batch], act=self.act_buf[batch],
                    mask=self.mask_buf[batch], rew=self.rew_buf[batch], ret=self.ret_buf[batch],
                    q_val=self.q_val_buf[batch])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}, batch
