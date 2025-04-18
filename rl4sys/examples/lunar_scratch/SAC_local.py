import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import copy

# -------------------------------------------------------------------------
# 1) Minimal MLP utility
# -------------------------------------------------------------------------
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

# -------------------------------------------------------------------------
# 2) Minimal abstract classes
# -------------------------------------------------------------------------
class ForwardKernelAbstract(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class StepKernelAbstract(nn.Module):
    def __init__(self):
        super().__init__()

    def step(self, *args, **kwargs):
        raise NotImplementedError

# -------------------------------------------------------------------------
# 3) Actor networks (ContinuousSAC, DiscreteSAC)
# -------------------------------------------------------------------------
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class ContinuousSAC(ForwardKernelAbstract):
    """
    Squashed gaussian MLP actor network.
    """
    def __init__(self, obs_dim, act_dim, act_limit, hidden_sizes=(256, 256),
                 activation=nn.ReLU, log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        super().__init__()
        self.actor_net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_limit = act_limit

    def _distribution(self, obs, mask=None):
        act = self.actor_net(obs)
        mu = self.mu_layer(act)
        log_std = self.log_std_layer(act)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return Normal(mu, std), mu

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, mask=None, deterministic=False, with_logprob=True):
        dist, mu = self._distribution(obs, mask)
        if deterministic:
            pi_action = mu
        else:
            pi_action = dist.sample()

        if with_logprob:
            logp_a = self._log_prob_from_distribution(dist, pi_action).sum(axis=-1)
            # Tanh-squash correction
            logp_a -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=-1)
        else:
            logp_a = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_a

class DiscreteSAC(ForwardKernelAbstract):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        self.actor_net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.logit_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.softmax = nn.Softmax(dim=-1)

    def _distribution(self, obs, mask=None):
        x = self.actor_net(obs)
        logits = self.logit_layer(x)
        probs = self.softmax(logits).clamp_min(1e-8)
        return Categorical(probs=probs), probs

    def _log_prob(self, probs):
        return torch.log(probs.clamp_min(1e-8))

    def forward(self, obs, mask=None, deterministic=False, with_logprob=True):
        dist, probs = self._distribution(obs, mask)
        if deterministic:
            # Argmax for discrete
            act = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            act = dist.sample().view(-1, 1)

        if with_logprob:
            logp_a = self._log_prob(probs)
        else:
            logp_a = None

        return act, probs, logp_a

# -------------------------------------------------------------------------
# 4) Q-functions (QFunction, DoubleQFunction, DoubleQActorCritic)
# -------------------------------------------------------------------------
class QFunction(ForwardKernelAbstract):
    """
    MLP Q-network.
    """
    def __init__(self, obs_dim, hidden_sizes, act_dim, activation, discrete=False, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.discrete = discrete

    def forward(self, obs, mask=None):
        return self.q(obs)

class DoubleQFunction(ForwardKernelAbstract):
    def __init__(self, obs_dim, hidden_sizes, act_dim, activation, discrete=False, seed=0):
        super().__init__()
        self.q1 = QFunction(obs_dim, hidden_sizes, act_dim, activation, discrete, seed+1)
        self.q2 = QFunction(obs_dim, hidden_sizes, act_dim, activation, discrete, seed+2)

    def forward(self, obs, mask=None):
        q1 = self.q1(obs, mask)
        q2 = self.q2(obs, mask)
        return q1, q2

class DoubleQActorCritic(StepKernelAbstract):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), activation=nn.ReLU,
                 log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX, discrete=False, seed=0):
        super().__init__()
        self.discrete = discrete

        # For continuous control: act_limit might be env.action_space.high
        # For discrete control: we might pass act_dim as "act_limit"
        act_limit = act_dim

        if self.discrete:
            self.pi = DiscreteSAC(obs_dim, act_dim, hidden_sizes, activation)
        else:
            self.pi = ContinuousSAC(obs_dim, act_dim, act_limit, hidden_sizes, activation,
                                    log_std_min, log_std_max)

        # Double Q networks
        self.q = DoubleQFunction(obs_dim, hidden_sizes, act_dim, activation, discrete, seed)

    def step(self, obs, mask=None, deterministic=False):
        with torch.no_grad():
            if self.discrete:
                a, _, _ = self.pi.forward(obs, mask, deterministic, True)
            else:
                a, _ = self.pi.forward(obs, mask, deterministic, True)
            return a, {}

# -------------------------------------------------------------------------
# 5) Replay Buffer
# -------------------------------------------------------------------------
class ReplayBuffer:
    """
    A simple FIFO replay buffer for SAC (discrete example).
    """
    def __init__(self, obs_dim, act_dim, size, gamma):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        # For discrete action: store single int action. If continuous, store float.
        self.act_buf = np.zeros((size,), dtype=np.int32)
        self.rew_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.gamma = gamma

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32),
                     obs2=torch.as_tensor(self.obs2_buf[idxs], dtype=torch.float32),
                     act=torch.as_tensor(self.act_buf[idxs], dtype=torch.long),
                     rew=torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32),
                     done=torch.as_tensor(self.done_buf[idxs], dtype=torch.float32))
        return batch

# -------------------------------------------------------------------------
# 6) The SAC Algorithm (discrete version for LunarLander)
# -------------------------------------------------------------------------
class SAC:
    def __init__(
        self, 
        obs_dim, 
        act_dim, 
        hidden_sizes=(256, 256), 
        gamma=0.99, 
        lr=3e-4,
        alpha=0.2, 
        polyak=0.995, 
        discrete=True
    ):
        self.gamma = gamma
        self.lr = lr
        self.alpha = alpha
        self.polyak = polyak
        self.discrete = discrete

        # Main model
        self.ac = DoubleQActorCritic(obs_dim, act_dim, hidden_sizes=hidden_sizes, discrete=discrete)
        # Target critic
        self.ac_targ = copy.deepcopy(self.ac.q)

        # Optimizers
        self.pi_params = self.ac.pi.parameters()
        self.q1_params = self.ac.q.q1.parameters()
        self.q2_params = self.ac.q.q2.parameters()
        self.pi_optimizer = Adam(self.pi_params, lr=self.lr)
        self.q1_optimizer = Adam(self.q1_params, lr=self.lr)
        self.q2_optimizer = Adam(self.q2_params, lr=self.lr)

    def select_action(self, obs, deterministic=False):
        obs_torch = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        act, _ = self.ac.step(obs_torch, deterministic=deterministic)
        # For discrete, act is shape [1, 1]. Get scalar.
        return int(act.item())

    def update(self, data):
        obs, obs2 = data['obs'], data['obs2']
        act, rew, done = data['act'], data['rew'], data['done']

        # === Compute target ===
        with torch.no_grad():
            _, probs_next, logp_next = self.ac.pi(obs2)
            q1_targ, q2_targ = self.ac_targ(obs2)
            min_q_targ = torch.min(q1_targ, q2_targ)
            V_next = (probs_next * (min_q_targ - self.alpha * logp_next)).sum(dim=1)
            backup = rew + self.gamma * (1 - done) * V_next

        # === Q1, Q2 loss ===
        q1, q2 = self.ac.q(obs)
        q1_act = q1.gather(1, act.view(-1,1)).squeeze()
        q2_act = q2.gather(1, act.view(-1,1)).squeeze()

        loss_q1 = 0.5 * ((q1_act - backup)**2).mean()
        loss_q2 = 0.5 * ((q2_act - backup)**2).mean()

        self.q1_optimizer.zero_grad()
        loss_q1.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        self.q2_optimizer.step()

        # Freeze Q-nets
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # === Pi loss ===
        _, probs_cur, logp_cur = self.ac.pi(obs)
        q1_cur, q2_cur = self.ac.q(obs)
        min_q_cur = torch.min(q1_cur, q2_cur)
        pi_loss = (probs_cur * (self.alpha * logp_cur - min_q_cur)).sum(dim=1).mean()

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-nets
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # === Polyak averaging for target Q networks
        with torch.no_grad():
            for p, p_targ in zip(self.ac.q.q1.parameters(), self.ac_targ.q1.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.ac.q.q2.parameters(), self.ac_targ.q2.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        return dict(LossQ1=loss_q1.item(), LossQ2=loss_q2.item(), LossPi=pi_loss.item())

# -------------------------------------------------------------------------
# 7) A minimal LunarLander training loop with TensorBoard logging
# -------------------------------------------------------------------------
def train_sac_lunarlander(
    episodes=500, 
    steps_per_episode=1000, 
    replay_size=100000, 
    batch_size=64
):
    env = gym.make("LunarLander-v2")
    obs_dim = env.observation_space.shape[0]  # 8
    act_dim = env.action_space.n              # 4
    sac_agent = SAC(obs_dim, act_dim, discrete=True)
    replay_buffer = ReplayBuffer(obs_dim, act_dim, replay_size, gamma=0.99)

    # Create a unique log directory with "SAC" in the folder name
    script_dir = os.path.dirname(os.path.abspath(__file__))  # directory of current .py file
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(script_dir, "logs", f"{timestamp}_SAC_logs")

    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to: {log_dir}")

    total_steps = 0
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        for t in range(steps_per_episode):
            act = sac_agent.select_action(obs, deterministic=False)
            next_obs, rew, done, _, _ = env.step(act)
            ep_return += rew

            replay_buffer.store(obs, act, rew, next_obs, done)
            obs = next_obs
            total_steps += 1

            # If there's enough data in buffer, do an update
            if replay_buffer.size > batch_size:
                batch = replay_buffer.sample_batch(batch_size)
                sac_agent.update(batch)

            if done:
                break

        # Log the episode reward to TensorBoard
        writer.add_scalar("EpisodeReturn", ep_return, ep)
        print(f"Episode {ep+1} | Return = {ep_return:.2f}")

    env.close()
    writer.close()

# -------------------------------------------------------------------------
# 8) Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    train_sac_lunarlander(episodes=200)
