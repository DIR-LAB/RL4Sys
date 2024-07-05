import numpy as np
import scipy.signal
import torch

"""
DQN Code
"""

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class ReplayBuffer:
    def __init__(self, obs_dim, mask_dim, buf_size, gamma, epsilon):
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

    def store(self, obs, act, mask, rew, q_val, next_obs):
        """

        Args:
            obs:
            act:
            mask:
            rew:
            next_obs:
            q_val:
        Returns:

        """
        index = self.ptr % self.max_size    # buffer index loops to lower values
        self.obs_buf[index] = obs
        self.act_buf[index] = act
        self.mask_buf[index] = mask
        self.rew_buf[index] = rew
        self.next_obs_buf[index] = next_obs
        self.q_val_buf[index] = q_val
        self.ptr += 1

    def finish_path(self, last_val=0):
        """

        Args:
            last_val:

        Returns:

        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.q_val_buf[path_slice], last_val)

        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self, batch_size):
        """
        
        """
        assert self.ptr < batch_size
        batch = np.random.choice(self.max_size, batch_size, replace=False)
        self.ptr, self.path_start_idx = 0, 0

        data = dict(obs=self.obs_buf[batch],
                    next_obs=self.next_obs_buf[batch], act=self.act_buf[batch],
                    mask=self.mask_buf[batch], ret=self.ret_buf[batch],
                    q_val=self.q_val_buf[batch])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}, batch
