from _common._algorithms.BaseReplayBuffer import combined_shape, discount_cumsum, ReplayBufferAbstract

import numpy as np
import random
import torch

"""
DQN Code
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
        self.relevant_version_buf = np.zeros(buf_size, dtype=np.bool_)
        self.ptr, self.full, self.max_size = 0, False, buf_size
        self.capacity = buf_size
        self.current_version = 1

    def store(self, obs, act, mask, rew, done, version):
        """
        Append one timestep of agent-environment interaction to the buffer.
        Stores this observation as the next observation of the previous transition.
        """
        assert self.ptr < self.max_size

        if version != self.current_version:
            self.current_version = version
            self.relevant_version_buf[:] = False

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.mask_buf[self.ptr] = mask
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.relevant_version_buf[self.ptr] = True
        # most accurate way to retrieve next observation, I imagine.
        if self.ptr == 0 and self.full:
            self.next_obs_buf[self.max_size - 1] = obs
        if self.ptr > 0:
            self.next_obs_buf[self.ptr - 1] = obs

        self.ptr += 1

        if self.ptr == self.capacity:
            self.ptr = 0
            self.full = True

    def finish_path(self, last_val=0):
        """
        Unused in DQN
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
                done: done flag

        """
        size = self.capacity if self.full else self.ptr
        assert self.ptr < self.max_size
        # random sample of indices
        current_version_indices = np.where(self.relevant_version_buf[:size])[0]

        assert len(current_version_indices) >= batch_size, "Not enough relevant samples for requested batch_size."

        batch = random.sample(current_version_indices.tolist(), batch_size)

        data = dict(obs=self.obs_buf[batch], next_obs=self.next_obs_buf[batch], act=self.act_buf[batch],
                    mask=self.mask_buf[batch], rew=self.rew_buf[batch],
                    done=self.done_buf[batch])

        return {k: torch.as_tensor(v) for k, v in data.items()}, batch
