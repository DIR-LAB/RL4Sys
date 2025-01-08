#!/usr/bin/env python3
"""
Merged PPO code for running on LunarLander-v2 with TensorBoard logging.
Each run will create logs in a unique folder named 'logs/PPO_<timestamp>'.

Usage:
  python ppo_lunarlander.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
import time
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

#########################################################
# 1) Base MLP and helper functions
#########################################################

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Build a feedforward neural network.
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    if np.isscalar(shape):
        return (length, shape)
    return (length, *shape)

def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sums of vectors.
    input: 
        vector x,
        [x0, x1, x2]
    output:
        [x0 + discount*x1 + discount^2*x2, x1 + discount*x2, x2]
    """
    out = np.zeros_like(x, dtype=np.float32)
    running_sum = 0
    for i in reversed(range(len(x))):
        running_sum = x[i] + discount*running_sum
        out[i] = running_sum
    return out

def statistics_scalar(x):
    """
    Return mean and std of x
    """
    x = np.array(x, dtype=np.float32)
    mean = x.mean()
    std = x.std()
    return mean, std

#########################################################
# 2) Abstract classes (light placeholders)
#########################################################

class ForwardKernelAbstract(nn.Module):
    """
    Minimal placeholder. In your original code, this is an abstract base class.
    """
    def __init__(self):
        super().__init__()

class StepKernelAbstract(nn.Module):
    """
    Minimal placeholder. In your original code, this is an abstract base class.
    """
    def __init__(self):
        super().__init__()

class ReplayBufferAbstract:
    """
    Minimal placeholder. In your original code, this is an abstract base class.
    """
    def __init__(self):
        pass

class AlgorithmAbstract:
    """
    Minimal placeholder. In your original code, this is an abstract base class.
    """
    def __init__(self):
        pass


#########################################################
# 3) Actor, Critic, ActorCritic
#########################################################

class RLActor(ForwardKernelAbstract):
    """Neural network of Actor (for discrete actions)."""

    def __init__(self, kernel_size: int, kernel_dim: int, act_dim: int, custom_network: nn.Sequential = None):
        super().__init__()
        if custom_network is None:
            self.pi_network = nn.Sequential(
                nn.Linear(kernel_dim*kernel_size, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, act_dim)
            )
        else:
            self.pi_network = custom_network

        self.kernel_size = kernel_size
        self.kernel_dim = kernel_dim
        self.act_dim = act_dim

    def _distribution(self, flattened_obs: torch.Tensor, mask: torch.Tensor) -> Categorical:
        """
        Returns a Categorical distribution over actions.
        `mask` is used to set invalid actions to very low logit (-inf).
        """
        x = self.pi_network(flattened_obs)
        logits = x + (mask - 1)*1e10  # mask=1 -> keep logit, mask=0 -> -inf
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi: torch.distributions.Distribution, act: torch.Tensor) -> torch.Tensor:
        return pi.log_prob(act)

    def forward(self, flattened_obs: torch.Tensor, mask: torch.Tensor, act: torch.Tensor = None):
        """
        Returns the distribution over actions and log_probs of given actions (if provided).
        """
        pi = self._distribution(flattened_obs, mask)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class RLCritic(ForwardKernelAbstract):
    """Neural network of Critic, producing an estimate for V(s)."""

    def __init__(self, obs_dim: int, hidden_sizes: tuple = (32, 16, 8),
                 activation: nn.Module = nn.ReLU, custom_network: nn.Sequential = None):
        super().__init__()
        self.obs_dim = obs_dim
        if custom_network is None:
            self.layer_sizes = [obs_dim] + list(hidden_sizes) + [1]
            self.activation = activation
            self.v_net = mlp(self.layer_sizes, self.activation)
        else:
            self.v_net = custom_network

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        mask is unused for the critic but kept for signature consistency.
        Returns a scalar (value function).
        """
        return torch.squeeze(self.v_net(obs), -1)


class RLActorCritic(StepKernelAbstract):
    """
    PPO Actor-Critic that holds an actor (pi) and critic (v).
    """

    def __init__(self, kernel_size: int, kernel_dim: int, act_dim: int):
        super().__init__()
        self.flatten_obs_dim = kernel_size * kernel_dim
        self.kernel_size = kernel_size
        self.kernel_dim = kernel_dim
        self.act_dim = act_dim

        # Actor
        self.pi = RLActor(kernel_size, kernel_dim, act_dim)
        # Critic
        self.v = RLCritic(self.flatten_obs_dim)

    def step(self, flattened_obs: torch.Tensor, mask: torch.Tensor):
        """
        Sample action from policy, get logp(a) and value estimate.
        Returns:
           a (numpy array): action index (discrete)
           dict: { 'v': state-value, 'logp_a': log probability }
        """
        with torch.no_grad():
            pi, _ = self.pi(flattened_obs, mask)
            a = pi.sample()  # action
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(flattened_obs, mask)
        return a.numpy(), {'v': v.numpy(), 'logp_a': logp_a.numpy()}

    def act(self, flattened_obs: torch.Tensor, mask: torch.Tensor):
        """
        Same as step, but only returns action.
        """
        with torch.no_grad():
            pi, _ = self.pi(flattened_obs, mask)
            action = pi.sample()
            return action.numpy()


#########################################################
# 4) Replay Buffer
#########################################################

class ReplayBuffer(ReplayBufferAbstract):
    """
    A buffer for storing trajectories for PPO with GAE-Lambda.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        super().__init__()
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size), dtype=np.float32)
        self.mask_buf = np.ones(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, mask, rew, val, logp):
        """
        Append one timestep.
        """
        assert self.ptr < self.max_size, "Buffer full!"
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.mask_buf[self.ptr] = mask
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Finish the trajectory. This uses GAE-Lambda for advantage calculation,
        and also computes reward-to-go for value function targets.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # Rewards-to-go
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Get all data from buffer, then normalize advantage, and reset.
        """
        assert self.ptr == self.path_start_idx, \
            "finish_path() must be called after the last trajectory!"

        data_size = self.ptr
        self.ptr, self.path_start_idx = 0, 0

        adv = self.adv_buf[:data_size]
        adv_mean, adv_std = statistics_scalar(adv)
        adv = (adv - adv_mean) / (adv_std + 1e-8)

        data = dict(
            obs = self.obs_buf[:data_size],
            act = self.act_buf[:data_size],
            mask = self.mask_buf[:data_size],
            ret = self.ret_buf[:data_size],
            adv = adv,
            logp = self.logp_buf[:data_size]
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


#########################################################
# 5) PPO Algorithm
#########################################################

class PPO(AlgorithmAbstract):
    """
    PPO Algorithm.  
    """

    def __init__(self, kernel_size: int, kernel_dim: int, act_dim:int,
                 buf_size: int, gamma=0.99, lam=0.95, clip_ratio=0.2,
                 pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80,
                 target_kl=0.01, writer=None):
        super().__init__()

        # Replay buffer
        obs_dim = kernel_size * kernel_dim
        self._replay_buffer = ReplayBuffer(obs_dim, act_dim, buf_size, gamma=gamma, lam=lam)

        # Actor-Critic
        self._model = RLActorCritic(kernel_size, kernel_dim, act_dim)

        # Hyperparams
        self._clip_ratio = clip_ratio
        self._train_pi_iters = train_pi_iters
        self._train_v_iters = train_v_iters
        self._target_kl = target_kl

        # Optimizers
        self._pi_optimizer = Adam(self._model.pi.parameters(), lr=pi_lr)
        self._vf_optimizer = Adam(self._model.v.parameters(), lr=vf_lr)

        # TensorBoard writer
        self.writer = writer

        # Book-keeping
        self.epoch = 0

    def save(self, filename: str) -> None:
        """
        Save entire model to disk.
        """
        torch.save(self._model.state_dict(), filename)

    def receive_trajectory(self, obs_list, act_list, mask_list, rew_list, val_list, logp_list, last_val=0):
        """
        Instead of RL4SysTrajectory, we feed lists of transitions for one episode.
        """
        for o, a, m, r, v, l in zip(obs_list, act_list, mask_list, rew_list, val_list, logp_list):
            self._replay_buffer.store(o, a, m, r, v, l)
        self._replay_buffer.finish_path(last_val)

    def train_model(self):
        """
        Run PPO update.
        """
        data = self._replay_buffer.get()

        # old pi loss
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy
        for i in range(self._train_pi_iters):
            self._pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self._target_kl:
                print(f"Early stopping at step {i} due to reaching max kl.")
                break
            loss_pi.backward()
            self._pi_optimizer.step()

        # Train value function
        for _ in range(self._train_v_iters):
            self._vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self._vf_optimizer.step()

        # Log info
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        pi_final = loss_pi.item()
        v_final = loss_v.item()
        print(f"[PPO Train] Epoch {self.epoch}, LossPi: {pi_l_old:.3f}->{pi_final:.3f}, "
              f"LossV: {v_l_old:.3f}->{v_final:.3f}, KL: {kl:.5f}, Ent: {ent:.3f}, ClipFrac: {cf:.3f}")

        # Write to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar("Loss/Policy", pi_final, self.epoch)
            self.writer.add_scalar("Loss/Value", v_final, self.epoch)
            self.writer.add_scalar("Policy/KL", kl, self.epoch)
            self.writer.add_scalar("Policy/Entropy", ent, self.epoch)
            self.writer.add_scalar("Policy/ClipFrac", cf, self.epoch)

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old, mask = data['obs'], data['act'], data['adv'], data['logp'], data['mask']
        # Policy
        pi, logp = self._model.pi(obs, mask, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)*adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # KL
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self._clip_ratio) | ratio.lt(1 - self._clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    def compute_loss_v(self, data):
        obs, ret, mask = data['obs'], data['ret'], data['mask']
        return ((self._model.v(obs, mask) - ret)**2).mean()


#########################################################
# 6) Minimal training loop on LunarLander-v2 + TensorBoard
#########################################################

def run_ppo_lunarlander(num_epochs=50, steps_per_epoch=4000, render=False):
    """
    Train PPO on LunarLander-v2 environment for a given number of epochs,
    each epoch collecting a total of 'steps_per_epoch' steps.
    """
    # Create log directory for TensorBoard
    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"PPO_{date_str}"
    log_dir = os.path.join("logs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved in: {log_dir}")

    env = gym.make("LunarLander-v2")
    obs_dim = env.observation_space.shape[0]   # e.g. 8
    act_dim = env.action_space.n               # e.g. 4 (discrete)

    # We'll treat the state as (kernel_size=1, kernel_dim=obs_dim)
    buf_size = steps_per_epoch

    # Instantiate PPO with writer
    ppo = PPO(kernel_size=1, kernel_dim=obs_dim, act_dim=act_dim, buf_size=buf_size,
              gamma=0.99, lam=0.95, clip_ratio=0.2,
              pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80,
              target_kl=0.01, writer=writer)

    total_steps = num_epochs * steps_per_epoch
    start_time = time.time()

    # Storage for a single trajectory
    obs_list, act_list, mask_list, rew_list, val_list, logp_list = [], [], [], [], [], []
    ep_ret, ep_len = 0, 0
    obs, _ = env.reset()

    # We'll keep a rolling count of completed episodes and their returns
    ep_count = 0
    returns_log = []

    for t in range(1, total_steps+1):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        mask = np.ones(act_dim, dtype=np.float32)  # all actions valid
        a, data = ppo._model.step(obs_tensor.view(1, -1), torch.as_tensor(mask).view(1, -1))

        action = a.item()
        value = data['v'].item()
        logp = data['logp_a'].item()

        next_obs, reward, done, _, _ = env.step(action)
        ep_ret += reward
        ep_len += 1

        # Save step
        obs_list.append(obs)
        act_list.append(action)
        mask_list.append(mask)
        rew_list.append(reward)
        val_list.append(value)
        logp_list.append(logp)

        obs = next_obs

        if render:
            env.render()

        if done:
            ep_count += 1
            returns_log.append(ep_ret)
            # Terminal state -> last_val=0
            ppo.receive_trajectory(obs_list, act_list, mask_list, rew_list, val_list, logp_list, last_val=0)
            obs_list, act_list, mask_list, rew_list, val_list, logp_list = [], [], [], [], [], []
            obs, _ = env.reset()
            ep_ret, ep_len = 0, 0

        # End of epoch -> train
        if t % steps_per_epoch == 0:
            ppo.epoch += 1
            # If partial trajectory left, finish it with last_val=0
            if len(obs_list) > 0:
                ppo.receive_trajectory(obs_list, act_list, mask_list, rew_list, val_list, logp_list, last_val=0)
                obs_list, act_list, mask_list, rew_list, val_list, logp_list = [], [], [], [], [], []

            # Train
            ppo.train_model()

            # Log average episode return to TensorBoard
            if len(returns_log) > 0:
                avg_return = np.mean(returns_log)
                writer.add_scalar("Return/Avg", avg_return, ppo.epoch)
                print(f"Epoch {ppo.epoch}, AvgEpRet: {avg_return:.3f}, EpCount: {ep_count}")
                returns_log = []  # reset

            print(f"Epoch {ppo.epoch} finished. Elapsed time: {time.time() - start_time:.2f} s")

    env.close()
    writer.close()
    print("Training complete! You can run `tensorboard --logdir logs` to view the results.")


if __name__ == "__main__":
    # Example run
    run_ppo_lunarlander(num_epochs=50, steps_per_epoch=4000, render=False)
