from _common._algorithms.BaseReplayBuffer import combined_shape, discount_cumsum, ReplayBufferAbstract

import numpy as np
import torch
import random
"""
PPO Code
"""


class ReplayBuffer(ReplayBufferAbstract):
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, mask_dim, buf_size, gamma=0.99, lam=0.95, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(buf_size), dtype=np.int32)
        self.mask_buf = np.zeros(combined_shape(buf_size, mask_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buf_size, dtype=np.float32)
        self.ret_buf = np.zeros(buf_size, dtype=np.float32)
        self.val_buf = np.zeros(buf_size, dtype=np.float32)
        self.logp_buf = np.zeros(buf_size, dtype=np.float32)
        self.adv_buf = np.zeros(buf_size, dtype=np.float32)
        self.done_buf = np.zeros(buf_size, dtype=np.bool_)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, buf_size
        self.capacity = buf_size

    def store(self, obs, act, mask, rew, val, logp, done):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        idx = self.ptr % self.max_size
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.mask_buf[idx] = mask
        self.rew_buf[idx] = rew
        self.val_buf[idx] = val
        self.logp_buf[idx] = logp
        self.done_buf[idx] = done

        if self.ptr > 0:
            self.next_obs_buf[idx - 1] = obs

        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, batch_size: int):
        """
        Get a batch of data from the buffer.
        """
        assert self.ptr >= batch_size
        # Random sample of indices
        batch = random.sample(range(len(self.obs_buf)), batch_size)

        data = dict(
            obs=self.obs_buf[batch],
            next_obs=self.next_obs_buf[batch],
            act=self.act_buf[batch],
            mask=self.mask_buf[batch],
            ret=self.ret_buf[batch],
            adv=self.adv_buf[batch],
            logp=self.logp_buf[batch],
            val=self.val_buf[batch],
            rew=self.rew_buf[batch],
            done=self.done_buf[batch]
        )

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}, batch
