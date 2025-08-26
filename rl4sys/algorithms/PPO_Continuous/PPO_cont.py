import os
import threading
import time
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from rl4sys.algorithms.PPO_Continuous.kernel import RLActorCriticCont
from rl4sys.common.trajectory import RL4SysTrajectory
from rl4sys.utils.util import StructuredLogger


class PPOCont():
    """
    PPO for continuous action spaces using a diagonal Gaussian policy and GAE.
    Mirrors the structure of the discrete `PPO` implementation.
    """

    def __init__(self,
                 version: int,
                 seed: int = 0,
                 input_size: int = 1,
                 act_dim: int = 1,
                 buf_size: int = 1000000,
                 batch_size: int = 64,
                 traj_per_epoch: int = 25600,
                 clip_ratio: float = 0.2,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 pi_lr: float = 3e-4,
                 vf_lr: float = 1e-3,
                 train_pi_iters: int = 10,
                 train_v_iters: int = 10,
                 target_kl: float = 0.015,
                 max_grad_norm: float = 0.5,
                 norm_adv: bool = True,
                 clip_vloss: bool = False,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 ):

        # Hyperparameters
        self._buf_size = buf_size
        self._batch_size = batch_size
        self._traj_per_epoch = traj_per_epoch
        self._clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam
        self._pi_lr = pi_lr
        self._vf_lr = vf_lr
        self._train_pi_iters = train_pi_iters
        self._train_v_iters = train_v_iters
        self._target_kl = target_kl
        self._max_grad_norm = max_grad_norm
        self._norm_adv = norm_adv
        self._clip_vloss = clip_vloss
        self._ent_coef = ent_coef
        self._vf_coef = vf_coef
        self.version = version
        self.seed = seed
        self.type = "onpolicy"
        self.act_dim = act_dim

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self._model_train = RLActorCriticCont(input_size, act_dim).to(self.device)

        self.models = {}
        self.lock = threading.RLock()
        self.models[self.version] = self._model_train

        # Optimizers
        self.actor_optimizer = Adam(self._model_train.pi.parameters(), lr=self._pi_lr)
        self.critic_optimizer = Adam(self._model_train.v.parameters(), lr=self._vf_lr)

        # Storage
        self.storage_obs = []
        self.storage_act = []
        self.storage_logp = []
        self.storage_rew = []
        self.storage_val = []
        self.storage_done = []
        self.storage_next_obs = []
        self.storage_version = []

        # Metrics
        self.ep_rewards = 0
        self.train_ep_rewards = 0
        self.start_time = None
        self.traj = 0
        self.epoch = 0
        self.global_step = 0

        # Logging
        self.logger = StructuredLogger(f"PPO-Cont-{version}", debug=True)
        log_data_dir = os.path.join('./logs/rl4sys-ppo-cont-info', f"{int(time.time())}__{self.seed}")
        os.makedirs(log_data_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_data_dir)
        self.save_model_path = os.path.join(log_data_dir, 'models')
        os.makedirs(self.save_model_path, exist_ok=True)

        self.logger.info("PPO-Cont initialized with RLActorCriticCont",
                        model_name=self._model_train.get_model_name(),
                        input_size=input_size,
                        act_dim=act_dim)

    def save(self, filename: str) -> None:
        new_path = os.path.join(self.save_model_path, filename + ('.pth' if not filename.__contains__('.pth') else ''))
        torch.save(self._model_train, new_path)

    def receive_trajectory(self, trajectory: RL4SysTrajectory, version: int) -> bool:
        # On-policy: ignore stale trajectories
        if version != self.version:
            self.logger.debug("Discarding stale trajectory", traj_version=version, current_version=self.version)
            return False

        if self.start_time is None:
            self.start_time = time.time()

        self.storage_version.append(version)
        for i, r4a in enumerate(trajectory):
            self.traj += 1
            self.global_step += 1
            self.ep_rewards += r4a.rew
            self.train_ep_rewards += r4a.rew

            obs_t = torch.as_tensor(r4a.obs, dtype=torch.float32).to(self.device)
            obs_value = self._model_train.get_value(obs_t)

            # Store
            self.storage_obs.append(np.copy(r4a.obs))
            self.storage_act.append(np.copy(r4a.act))  # continuous vector
            self.storage_logp.append(np.copy(r4a.data['logp_a']))
            self.storage_rew.append(np.copy(r4a.rew))
            self.storage_val.append(np.copy(obs_value.detach().cpu().numpy()))
            self.storage_done.append(r4a.done)

            if i < len(trajectory) - 1:
                self.storage_next_obs.append(trajectory[i + 1].obs)
            else:
                if not r4a.done:
                    self.storage_next_obs.append(r4a.obs)
                else:
                    self.storage_next_obs.append(np.zeros_like(r4a.obs))

            self.writer.add_scalar('charts/VVals', obs_value, self.global_step)

            if r4a.done:
                self.writer.add_scalar("charts/reward", self.ep_rewards, self.epoch)
                self.ep_rewards = 0

        # Update when enough transitions accumulated
        if self.traj > 0 and self.traj > self._traj_per_epoch:
            print(f"\n-----[PPO-Cont] Training model for epoch {self.epoch}-----\n")
            pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, explained_var = self.train_model()
            self.epoch += 1
            self.writer.add_scalar("losses/pg_loss", pg_loss, self.global_step)
            self.writer.add_scalar("losses/v_loss", v_loss, self.global_step)
            self.writer.add_scalar("losses/entropy_loss", entropy_loss, self.global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl, self.global_step)
            self.writer.add_scalar("losses/clipfracs", clipfracs, self.global_step)
            self.writer.add_scalar("losses/explained_var", explained_var, self.global_step)
            self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
            self.writer.add_scalar("charts/avg_reward", self.train_ep_rewards/100, self.epoch)
            self.train_ep_rewards = 0
            self.logger.info("PPO-Cont training epoch completed",
                            epoch=self.epoch, pg_loss=pg_loss, v_loss=v_loss,
                            entropy_loss=entropy_loss, approx_kl=approx_kl,
                            clipfracs=clipfracs, explained_var=explained_var)
            return True
        return False

    def get_current_model(self):
        with self.lock:
            return self.models[self.version], self.version

    def _clear_storage(self):
        self.storage_obs = []
        self.storage_act = []
        self.storage_logp = []
        self.storage_rew = []
        self.storage_val = []
        self.storage_done = []
        self.storage_next_obs = []
        self.storage_version = []

    def train_model(self):
        # To numpy
        obs = np.array(self.storage_obs, dtype=np.float32)
        actions = np.array(self.storage_act, dtype=np.float32)  # continuous
        logprobs = np.array(self.storage_logp, dtype=np.float32)
        rewards = np.array(self.storage_rew, dtype=np.float32)
        values = np.array(self.storage_val, dtype=np.float32)
        dones = np.array(self.storage_done, dtype=np.bool_)
        next_obs = np.array(self.storage_next_obs, dtype=np.float32)

        # To tensors
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32).to(self.device)
        logprobs_tensor = torch.tensor(logprobs, dtype=torch.float32).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(self.device)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(self.device)

        # GAE
        with torch.no_grad():
            next_values = self._model_train.get_value(next_obs_tensor)
            advantages = torch.zeros_like(rewards_tensor)
            lastgaelam = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones_tensor[t].float()
                    nextvalues = next_values[t]
                else:
                    nextnonterminal = 1.0 - dones_tensor[t].float()
                    nextvalues = next_values[t]
                delta = rewards_tensor[t] + self.gamma * nextvalues * nextnonterminal - values_tensor[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            returns = advantages + values_tensor

        # Flatten
        b_obs = obs_tensor.reshape(-1, obs.shape[-1])
        b_actions = actions_tensor.reshape(-1, actions.shape[-1])
        b_logprobs = logprobs_tensor.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_tensor.reshape(-1)

        if self._norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        clipfracs = []

        # Policy update
        for _ in range(self._train_pi_iters):
            pi_dist = self._model_train.pi._distribution(b_obs)
            new_logp = pi_dist.log_prob(b_actions)
            logratio = new_logp - b_logprobs
            ratio = torch.exp(logratio)

            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio) * b_advantages
            entropy = pi_dist.entropy()
            pg_loss = -torch.mean(torch.min(surr1, surr2)) - self._ent_coef * torch.mean(entropy)

            approx_kl = torch.mean(b_logprobs - new_logp).item()
            clipfracs.append(((ratio - 1.0).abs() > self._clip_ratio).float().mean().item())

            self.actor_optimizer.zero_grad()
            pg_loss.backward()
            if self._max_grad_norm and self._max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self._model_train.pi.parameters(), self._max_grad_norm)
            self.actor_optimizer.step()

            if self._target_kl is not None and approx_kl > 1.5 * self._target_kl:
                break

        entropy_loss = pi_dist.entropy().mean().item()

        # Value update
        for _ in range(self._train_v_iters):
            value_pred = self._model_train.get_value(b_obs).view(-1)
            if self._clip_vloss:
                v_loss_unclipped = (value_pred - b_returns) ** 2
                v_clipped = b_values + torch.clamp(value_pred - b_values, -self._clip_ratio, self._clip_ratio)
                v_loss_clipped = (v_clipped - b_returns) ** 2
                v_loss = 0.5 * torch.mean(torch.max(v_loss_unclipped, v_loss_clipped))
            else:
                v_loss = torch.mean((b_returns - value_pred) ** 2)

            v_loss = self._vf_coef * v_loss
            self.critic_optimizer.zero_grad()
            v_loss.backward()
            if self._max_grad_norm and self._max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self._model_train.v.parameters(), self._max_grad_norm)
            self.critic_optimizer.step()

        # Explained variance
        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Bump version and clear storage
        with self.lock:
            self.version += 1
            self.models[self.version] = copy.deepcopy(self._model_train)
            self._clear_storage()
            self.traj = 0

        return pg_loss.item(), v_loss.item(), entropy_loss, approx_kl, float(np.mean(clipfracs)) if clipfracs else 0.0, explained_var


