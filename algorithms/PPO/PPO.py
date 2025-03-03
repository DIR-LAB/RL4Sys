from _common._algorithms.BaseAlgorithm import AlgorithmAbstract

import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn

from .kernel import RLActorCritic
from .replay_buffer import ReplayBuffer

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import EpochLogger, setup_logger_kwargs
from protocol.trajectory import RL4SysTrajectory

from utils.conf_loader import ConfigLoader
"""Import and load RL4Sys/config.json PPO algorithm configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
config_loader = ConfigLoader(algorithm='PPO')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path


class PPO(AlgorithmAbstract):
    """
    A refactored PPO class inspired by ppo_cleanrl.py

    Key changes:
    - Single Adam optimizer for both actor and critic (self.optimizer).
    - GAE-Lambda advantage calculations done in 'compute_advantages()'.
    - We store all transitions in the replay buffer. On 'traj_per_epoch' triggers,
      we compute advantages/returns and do a PPO update.
    """

    def __init__(self, env_dir: str,
                 input_size: int,
                 act_dim: int,
                 buf_size: int,
                 batch_size: int = hyperparams['batch_size'],
                 seed: int = hyperparams['seed'],
                 traj_per_epoch: int = hyperparams['traj_per_epoch'],
                 clip_ratio: float = hyperparams['clip_ratio'],
                 gamma: float = hyperparams['gamma'],
                 lam: float = hyperparams['lam'],
                 pi_lr: float = hyperparams['pi_lr'],
                 vf_lr: float = hyperparams['vf_lr'],
                 train_pi_iters: int = hyperparams['train_pi_iters'],
                 train_v_iters: int = hyperparams['train_v_iters'],
                 target_kl: float = hyperparams['target_kl'],
                 max_grad_norm: float = 0.5,
                 norm_adv: bool = True,
                 clip_vloss: bool = True,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 ):

        super().__init__()
        seed += 10000 * os.getpid()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Hyperparameters
        self._traj_per_epoch = traj_per_epoch
        self._clip_ratio = clip_ratio
        self._batch_size = batch_size
        self._train_pi_iters = train_pi_iters
        self._train_v_iters = train_v_iters
        self._target_kl = target_kl
        self._max_grad_norm = max_grad_norm
        self._norm_adv = norm_adv
        self._clip_vloss = clip_vloss
        self._ent_coef = ent_coef
        self._vf_coef = vf_coef
        self.gamma = gamma
        self.lam = lam

        # Create a single actor-critic "kernel" like ppo_cleanrl
        self._model_train = RLActorCritic(input_size, act_dim)
        self._model = self._model_train.actor  # for usage consistency

        # Single Adam optimizer for both policy and value function
        self.optimizer = Adam(self._model_train.parameters(), lr=pi_lr, eps=1e-5)

        # set up logger
        log_data_dir = os.path.join(env_dir, './logs/')
        logger_kwargs = setup_logger_kwargs("rl4sys-ppo-info", seed=seed, data_dir=log_data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.logger.setup_pytorch_saver(self._model_train)

        # Buffers for storing one "epoch" worth of transitions
        # For simplicity, we store in lists, then convert to Tensors
        self.storage_obs = []
        self.storage_act = []
        self.storage_logp = []
        self.storage_rew = []
        self.storage_val = []
        self.storage_done = []

        self.traj = 0
        self.epoch = 0

    def save(self, filename: str) -> None:
        """Save model as file.

        Uses .pth file extension.

        Args:
            filename: name to save file as

        """
        new_path = os.path.join(save_model_path, filename +
                                ('.pth' if not filename.__contains__('.pth') else ''))
        torch.save(self._model_train, new_path)

    def receive_trajectory(self, trajectory: RL4SysTrajectory) -> bool:
        """
        Process a trajectory from the environment. For each step:
         - store (obs, act, logp_a, rew, val, done) in local buffers
        If an epoch is triggered, calls train_model().

        Returns True if we just finished an epoch (implies new model).
        """
        self.traj += 1
        ep_ret, ep_len = 0, 0

        for r4a in trajectory:
            ep_ret += r4a.rew
            ep_len += 1

            if not r4a.done:
                self.storage_obs.append(r4a.obs)
                self.storage_act.append(r4a.act)
                self.storage_logp.append(r4a.data['logp_a'])
                self.storage_rew.append(r4a.rew)
                self.storage_val.append(r4a.data['v'])
                self.storage_done.append(False)
                self.logger.store(VVals=r4a.data['v'])
            else:
                # Mark a final 'done' transition for advantage logic
                self.storage_obs.append(r4a.obs if r4a.obs is not None else np.zeros_like(self.storage_obs[-1]))
                self.storage_act.append(r4a.act if r4a.act is not None else np.zeros_like(self.storage_act[-1]))
                self.storage_logp.append(r4a.data['logp_a'] if r4a.data else 0.0)
                self.storage_rew.append(r4a.rew if r4a.rew is not None else 0)
                self.storage_val.append(r4a.data['v'] if (r4a.data and 'v' in r4a.data) else 0)
                self.storage_done.append(True)

                # Log episode stats
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)

        # Once we have enough 'trajectories', do an update like ppo_cleanrl
        if self.traj > 0 and self.traj % self._traj_per_epoch == 0:
            self.epoch += 1
            self.train_model()
            self.log_epoch()
            return True  # indicates an updated model
        return False

    def train_model(self) -> None:
        """
        Replaces the old train_model with a single method that:
         1) Converts stored transitions to Tensors
         2) Computes advantages/returns via GAE-Lambda (with proper bootstrapping)
         3) Performs PPO update (policy and value) in mini-batches
         4) Clears the storage buffers for next epoch
        """
        # 1) Convert stored transitions to Tensors
        obs_arr = np.array(self.storage_obs, dtype=np.float32)
        act_arr = np.array(self.storage_act, dtype=np.int64)  # discrete actions
        logp_arr = np.array(self.storage_logp, dtype=np.float32)
        rew_arr = np.array(self.storage_rew, dtype=np.float32)
        val_arr = np.array(self.storage_val, dtype=np.float32)
        done_arr = np.array(self.storage_done, dtype=np.bool_)

        # Compute bootstrap value for the final state if not terminal.
        if not self.storage_done[-1]:
            last_obs = torch.as_tensor(self.storage_obs[-1], dtype=torch.float32).unsqueeze(0)
            last_value = self._model_train.get_value(last_obs).detach().cpu().numpy()[0]
        else:
            last_value = 0.0

        # 2) Compute advantages using GAE-Lambda with bootstrapping (fixed)
        advantages, returns = self.compute_advantages(
            rew_arr, val_arr, done_arr, gamma=self.gamma, lam=self.lam, last_value=last_value
        )

        # Convert everything to Torch
        obs_tensor = torch.as_tensor(obs_arr)
        act_tensor = torch.as_tensor(act_arr)
        logp_old_tensor = torch.as_tensor(logp_arr)
        adv_tensor = torch.as_tensor(advantages)
        ret_tensor = torch.as_tensor(returns)
        val_old_tensor = torch.as_tensor(val_arr)

        # 3) PPO update (like cleanrl) in mini-batches
        dataset_size = len(obs_tensor)
        indices = np.arange(dataset_size)
        mini_batch_size = max(1, self._batch_size)
        n_updates = self._train_pi_iters  # unified policy and value updates

        for _ in range(n_updates):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = indices[start:end]

                batch_obs = obs_tensor[mb_inds]
                batch_act = act_tensor[mb_inds]
                batch_adv = adv_tensor[mb_inds]
                batch_ret = ret_tensor[mb_inds]
                batch_logp_old = logp_old_tensor[mb_inds]
                batch_val_old = val_old_tensor[mb_inds]

                # Normalize advantages if desired
                if self._norm_adv:
                    # Normalize advantages if desired
                    batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std(unbiased=False) + 1e-8)


                # Forward pass using updated model
                _, logp_new, ent, v_new = self._model_train.step(batch_obs, action=batch_act)

                ratio = torch.exp(logp_new - batch_logp_old)
                pg_loss1 = -batch_adv * ratio
                pg_loss2 = -batch_adv * torch.clamp(
                    ratio,
                    1.0 - self._clip_ratio,
                    1.0 + self._clip_ratio
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if self._clip_vloss:
                    v_loss_unclipped = (v_new - batch_ret) ** 2
                    v_clipped = batch_val_old + torch.clamp(
                        v_new - batch_val_old,
                        -self._clip_ratio,
                        self._clip_ratio
                    )
                    v_loss_clipped = (v_clipped - batch_ret) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((v_new - batch_ret) ** 2).mean()

                ent_loss = ent.mean()
                loss = pg_loss - self._ent_coef * ent_loss + self._vf_coef * v_loss

                approx_kl = 0.5 * ((logp_new - batch_logp_old) ** 2).mean().item()
                if self._target_kl and approx_kl > 1.5 * self._target_kl:
                    break

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model_train.parameters(), self._max_grad_norm)
                self.optimizer.step()

        self.logger.store(
            LossPi=pg_loss.item(),
            LossV=v_loss.item(),
            Entropy=ent_loss.item(),
            KL=approx_kl
        )

        # 4) Clear our local storage
        del self.storage_obs[:]
        del self.storage_act[:]
        del self.storage_logp[:]
        del self.storage_rew[:]
        del self.storage_val[:]
        del self.storage_done[:]

    def compute_advantages(self, rewards, values, dones, gamma, lam, last_value):
        """
        Compute advantages using GAE-Lambda with proper bootstrapping for non-terminal episodes.

        Args:
            rewards: array of rewards.
            values: array of value estimates.
            dones: array indicating terminal transitions.
            gamma: discount factor.
            lam: GAE lambda.
            last_value: bootstrap value for the final state if not terminal.
        Returns:
            advantages and returns.
        """
        length = len(rewards)
        advantages = np.zeros(length, dtype=np.float32)
        lastgaelam = 0.0
        
        for t in reversed(range(length)):
            if t == length - 1:
                next_nonterminal = 1.0 - float(dones[t])
                next_value = last_value
            else:
                next_nonterminal = 1.0 - float(dones[t+1])
                next_value = values[t+1]
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam

        returns = advantages + values
        return advantages, returns

    def log_epoch(self) -> None:
        """Log the information collected in logger over the course of the last epoch.
        """
        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', with_min_and_max=True)
        self.logger.log_tabular('VVals', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.log_tabular('KL', average_only=True)
        self.logger.dump_tabular()
