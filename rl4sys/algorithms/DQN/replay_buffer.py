import numpy as np
import random
import torch
import numpy as np
import scipy.signal

"""
DQN Code
"""

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum = np.sum(x)
    global_n = len(x)
    mean = global_sum / global_n

    global_sum_sq = (np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = np.min(x) if len(x) > 0 else np.inf
        global_max = np.max(x) if len(x) > 0 else -np.inf
        return mean, std, global_min, global_max
    return mean, std


class ReplayBuffer():
    def __init__(self, obs_dim, buf_size, gamma, epsilon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(buf_size), dtype=np.int32)
        self.rew_buf = np.zeros(buf_size, dtype=np.float32)
        self.ret_buf = np.zeros(buf_size, dtype=np.float32)
        self.q_val_buf = np.zeros(buf_size, dtype=np.float32)
        self.done_buf = np.zeros(buf_size, dtype=np.bool_)  # or np.float32
        self.gamma, self.epsilon = gamma, epsilon
        self.ptr, self.path_start_idx, self.max_size = 0, 0, buf_size
        self.capacity = buf_size


    def store(self, obs, act, rew, q_val, done):
        idx = self.ptr % self.max_size
        self.obs_buf[idx] = obs
        self.act_buf[idx] = act
        self.rew_buf[idx] = rew
        self.done_buf[idx] = done
        self.q_val_buf[idx] = q_val[int(act)] if len(q_val) > act else 0.0 # TODO check if this is correct

        if self.ptr > 0:
            self.next_obs_buf[idx - 1] = obs

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
                    rew=self.rew_buf[batch], ret=self.ret_buf[batch],
                    q_val=self.q_val_buf[batch], done=self.done_buf[batch])

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}, batch
