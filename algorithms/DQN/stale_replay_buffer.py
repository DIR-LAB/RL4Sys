from _common._algorithms.BaseReplayBuffer import combined_shape, discount_cumsum, ReplayBufferAbstract

import numpy as np
import random
import torch

"""
DQN Code
"""


class StaleReplayBuffer(ReplayBufferAbstract):
    def __init__(self, obs_dim, mask_dim, buf_size,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(buf_size), dtype=np.int32)
        self.mask_buf = np.zeros(combined_shape(buf_size, mask_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buf_size, dtype=np.float32)
        self.done_buf = np.zeros(buf_size, dtype=np.bool_)
        self.stale_sample_marker_buf = np.zeros(buf_size, dtype=np.int32)
        self.last_traj_before_training = -1
        self.ptr, self.full, self.max_size = 0, False, buf_size
        self.capacity = buf_size

    def store(self, obs, act, mask, rew, done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        Stores this observation as the next observation of the previous transition.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.mask_buf[self.ptr] = mask
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.stale_sample_marker_buf[self.ptr] = -1
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
        Call this at the end of a trajectory, or when one gets cut off by an epoch ending.
        Looks back in buffer to where the trajectory started, and uses the rewards found there to
        calculate the reward-to-go for each state in the trajectory.
        """
        pass

    def set_training_markers(self):
        if self.full:
            self.stale_sample_marker_buf[self.ptr:] += 1
            # set the latest + last sample as stale
            self.stale_sample_marker_buf[:self.ptr] = 1
        else:
            self.stale_sample_marker_buf[:self.ptr] =1
        if self.last_traj_before_training == -1:
            self.last_traj_before_training = self.ptr

    def get(self, batch_size: int, alpha: float = .95, beta: float = 0.005):
        """ Sample a batch of data from the replay buffer.

        Args:
            batch_size: the number of samples to draw
            alpha: the exponent of the age-based importance sampling weights

        Returns:
            A dictionary containing the following keys:
                obs: the current observation
                next_obs: the next observation
                act: the action
                rew: the reward
                ret: the reward-to-go
        """
        size = self.max_size if self.full else self.ptr
        assert self.ptr < self.max_size
        assert size >= batch_size

        ages = self.stale_sample_marker_buf[:size]

        indices = np.arange(size)
        weights = np.exp(-alpha * ages)
        weights /= np.sum(weights)

        # random sample of indices
        batch = np.random.choice(indices, size=batch_size, replace=False, p=weights)

        data = dict(obs=self.obs_buf[batch], next_obs=self.next_obs_buf[batch], act=self.act_buf[batch],
                    mask=self.mask_buf[batch], rew=self.rew_buf[batch],
                    done=self.done_buf[batch])

        loss_weights = torch.as_tensor(np.exp(-beta * ages[batch]), dtype=torch.float32)
        loss_weights = loss_weights / (loss_weights.sum() + 1e-8)
        batch_age = torch.as_tensor(loss_weights, dtype=torch.float32)

        return {k: torch.as_tensor(v) for k, v in data.items()}, batch, batch_age
