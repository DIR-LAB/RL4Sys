from _common._algorithms.BaseReplayBuffer import combined_shape, ReplayBufferAbstract
import numpy as np
import torch
import random

class ReplayBuffer(ReplayBufferAbstract):
    def __init__(self, obs_dim, act_dim, buf_size, gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(buf_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buf_size, dtype=np.float32)
        self.done_buf = np.zeros(buf_size, dtype=np.bool_)
        
        self.gamma = gamma
        self.ptr, self.size, self.max_size = 0, 0, buf_size

    def store(self, obs, act, rew, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        if self.ptr > 0:
            self.next_obs_buf[self.ptr - 1] = obs
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        data = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs]
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}, idxs
