from _common._algorithms.BaseReplayBuffer import combined_shape, discount_cumsum, ReplayBufferAbstract

import numpy as np
import random
import torch

"""
DDPG Code
"""


class ReplayBuffer(ReplayBufferAbstract):
    def __init__(self, obs_dim, mask_dim, buf_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(buf_size), dtype=np.int32)
        self.mask_buf = np.zeros(combined_shape(buf_size, mask_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buf_size, dtype=np.float32)
        self.done_buf = np.zeros(buf_size, dtype=np.bool_)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, buf_size
        self.capacity = buf_size

    def store(self, obs, next_obs, act, mask, rew, done):
        # Use the same index for both obs and next_obs
        idx = self.ptr % self.max_size  # or whatever indexing logic you like

        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.act_buf[idx] = act
        self.mask_buf[idx] = mask
        self.rew_buf[idx] = rew
        self.done_buf[idx] = done

        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Unused in DDPG
        """
        pass

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
        assert self.ptr >= batch_size
        # random sample of indices
        batch = random.sample(range(len(self.obs_buf)), batch_size)
        # self.ptr, self.path_start_idx = 0, 0 # TODO debug try use all traj, not first 32

        data = dict(obs=self.obs_buf[batch], next_obs=self.next_obs_buf[batch], act=self.act_buf[batch],
                    mask=self.mask_buf[batch], rew=self.rew_buf[batch],
                    done=self.done_buf[batch])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}, batch
