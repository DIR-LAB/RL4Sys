from _common._algorithms.BaseReplayBuffer import combined_shape, discount_cumsum, ReplayBufferAbstract

import numpy as np
import random
import torch

"""
DQN Code
"""


class ReplayBuffer(ReplayBufferAbstract):
    def __init__(self, obs_dim, mask_dim, buf_size, gamma, epsilon, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    """
    def store(self, obs, act, mask, rew, q_val):
        assert self.ptr <= self.max_size
        if self.ptr < self.max_size:
            self.obs_buf[self.ptr] = obs
            self.act_buf[self.ptr] = act
            self.mask_buf[self.ptr] = mask
            self.rew_buf[self.ptr] = rew
            self.q_val_buf[self.ptr] = q_val
            # most accurate way to retrieve next observation, I imagine.
            if self.ptr > 0:
                self.next_obs_buf[self.ptr - 1] = obs
            self.ptr += 1
        else:
            # Buffer is full: remove the oldest trajectory and shift everything left by one.
            # By doing this, index 0 will always be the oldest entry, and ptr remains max_size.

            # Shift all buffer contents one step to the left
            self.obs_buf[:-1] = self.obs_buf[1:]
            self.act_buf[:-1] = self.act_buf[1:]
            self.mask_buf[:-1] = self.mask_buf[1:]
            self.rew_buf[:-1] = self.rew_buf[1:]
            self.q_val_buf[:-1] = self.q_val_buf[1:]
            self.next_obs_buf[:-1] = self.next_obs_buf[1:]

            # Insert the new trajectory at the last position
            last_idx = self.max_size - 1
            self.obs_buf[last_idx] = obs
            self.act_buf[last_idx] = act
            self.mask_buf[last_idx] = mask
            self.rew_buf[last_idx] = rew
            self.q_val_buf[last_idx] = q_val

            # Update the next_obs for the second-to-last entry
            if self.ptr > 0:
                self.next_obs_buf[last_idx - 1] = obs
    """
    def store(self, obs, next_obs, act, mask, rew, q_val):
        # Use the same index for both obs and next_obs
        idx = self.ptr % self.max_size  # or whatever indexing logic you like

        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.act_buf[idx] = act
        self.mask_buf[idx] = mask
        self.rew_buf[idx] = rew
        self.q_val_buf[idx] = np.max(q_val)

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
        assert self.ptr >= batch_size
        # random sample of indices
        batch = random.sample(range(len(self.obs_buf)), batch_size)
        # self.ptr, self.path_start_idx = 0, 0 # TODO debug try use all traj, not first 32

        data = dict(obs=self.obs_buf[batch], next_obs=self.next_obs_buf[batch], act=self.act_buf[batch],
                    mask=self.mask_buf[batch], rew=self.rew_buf[batch], ret=self.ret_buf[batch],
                    q_val=self.q_val_buf[batch])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}, batch
