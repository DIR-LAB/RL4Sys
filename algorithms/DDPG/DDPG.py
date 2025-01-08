from _common._algorithms.BaseAlgorithm import AlgorithmAbstract

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .kernel import ActorCritic
from .replay_buffer import ReplayBuffer

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import EpochLogger, setup_logger_kwargs
from trajectory import RL4SysTrajectory

from conf_loader import ConfigLoader

import zmq

"""
Import and load RL4Sys/config.json DDPG Agent configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
config_loader = ConfigLoader(algorithm='DDPG')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path


class DDPG(AlgorithmAbstract):
    def __init__(self, env_dir: str, kernel_size: int, kernel_dim: int, act_dim: int, buf_size: int,
                 batch_size: int = hyperparams['batch_size'], seed: int = hyperparams['seed'],
                 traj_per_epoch: int = hyperparams['traj_per_epoch'], gamma: float = hyperparams['gamma'],
                 polyak: float = hyperparams['polyak'], act_noise_std: float = hyperparams['act_noise_std'],
                 act_noise_clip: float = hyperparams['act_noise_clip'], pi_lr: float = hyperparams['pi_lr'],
                 q_lr: float = hyperparams['q_lr'], train_iters: int = hyperparams['train_iters'],
                 target_update_freq: int = hyperparams['target_update_freq']):
        super().__init__()
        seed += 10000 * os.getpid()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Input parameters
        self._kernel_size = kernel_size
        self._kernel_dim = kernel_dim
        self._buf_size = buf_size
        self._act_dim = act_dim
        self._batch_size = batch_size

        # Hyperparameters
        self._traj_per_epoch = traj_per_epoch
        self._gamma = gamma
        self._polyak = polyak
        self._act_noise_std = act_noise_std
        self._act_noise_clip = act_noise_clip
        self._train_iters = train_iters
        self._target_update_freq = target_update_freq

        self._replay_buffer = ReplayBuffer(kernel_size*kernel_dim, act_dim, buf_size)

        self._model = ActorCritic(kernel_size*kernel_dim, act_dim, [256, 256], nn.ReLU, act_dim)
        self._target_model = ActorCritic(kernel_size*kernel_dim, act_dim, [256, 256], nn.ReLU, act_dim)
        self._target_model.load_state_dict(self._model.state_dict())
        for param in self._target_model.parameters():
            param.requires_grad = False

        self._pi_optimizer = Adam(self._model.pi.parameters(), lr=pi_lr)
        self._q_optimizer = Adam(self._model.v.parameters(), lr=q_lr)

        log_data_dir = os.path.join(env_dir, './logs/')
        logger_kwargs = setup_logger_kwargs("rl4sys-ddpg-info", seed=seed, data_dir=log_data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.logger.setup_pytorch_saver(self._model)

        self.traj = 0
        self.epoch = 0

        self.total_steps = 0

    def save(self, filename: str) -> None:
        """Save model as file.

        Uses .pth file extension.

        Args:
            filename: name to save file as

        """
        new_path = os.path.join(save_model_path, filename +
                                ('.pth' if not filename.__contains__('.pth') else ''))
        torch.save(self._model, new_path)

    def receive_trajectory(self, trajectory: RL4SysTrajectory) -> bool:
        """Process a trajectory received by training_server.

        If an epoch is triggered, calls train_model().

        Args:
            trajectory: holds agent experiences since last trajectory
        Returns:
            True if an epoch was triggered and an updated model should be sent.

        """
        self.traj += 1
        ep_ret, ep_len = 0, 0

        for r4a in trajectory.actions:
            ep_ret += r4a.rew
            ep_len += 1
            if not r4a.done:
                self._replay_buffer.store(r4a.obs, r4a.next_obs, r4a.act, r4a.mask, r4a.rew)
                self.logger.store(QVals=r4a.data['q_val'])
            else:
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)

        if self.traj > 0 and self.traj % self._traj_per_epoch == 0:
            self.epoch += 1
            self.train_model()
            self.log_epoch()
            return True

        return False

    def train_model(self) -> None:
        """Train model on data from DDPG replay_buffer.
        """
        data, batch = self._replay_buffer.get(self._batch_size)

        pi_l_old = self.compute_loss_pi(data)[0]
        q_l_old = self.compute_loss_q(data)[0]

        for i in range(self._train_iters):
            self._q_optimizer.zero_grad()
            loss_q, q_target = self.compute_loss_q(data)
            loss_q.backward()
            self._q_optimizer.step()

            self._pi_optimizer.zero_grad()
            loss_pi, pi_target = self.compute_loss_pi(data)
            loss_pi.backward()
            self._pi_optimizer.step()

            self.total_steps += 1
            if self.total_steps % self._target_update_freq == 0:
                # update policy network
                for param, target_param in zip(self._model.actor.parameters(), self._target_model.actor.parameters()):
                    target_param.data.copy_(self._polyak * param.data + (1 - self._polyak) * target_param.data)
                # update q network
                for param, target_param in zip(self._model.q_critic.parameters(), self._target_model.q_critic.parameters()):
                    target_param.data.copy_(self._polyak * param.data + (1 - self._polyak) * target_param.data)

        self.logger.store(StopIter=i)
        self.logger.store(QTargets=q_target, LossQ=loss_q, LossPi=loss_pi, DeltaLossQ=(loss_q - q_l_old),
                          DeltaLossPi=(loss_pi - pi_l_old))

    def log_epoch(self) -> None:
        """Log the information collected in logger over the course of the last epoch
        """
        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('QVals', average_only=True)
        self.logger.log_tabular('QTargets', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('DeltaLossPi', average_only=True)
        self.logger.log_tabular('DeltaLossQ', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)
        self.logger.dump_tabular()

    def compute_loss_pi(self, data):
        obs, mask, act, rew, next_obs = data['obs'], data['mask'], data['act'], data['rew'], data['next_obs']

        pi = self._model.actor.forward(obs, mask)
        pi_q = self._model.q_critic.forward(obs, mask, pi).mean()

        return pi_q

    def compute_loss_q(self, data):
        obs, mask, act, rew, next_obs, done = data['obs'], data['mask'], data['act'], data['rew'], data['next_obs'], data['done']

        with torch.no_grad():
            act_target = self._target_model.actor.forward(next_obs, mask)
            q_target = self._target_model.q_critic.forward(next_obs, mask, act_target)
            targets = rew + (self._gamma * (1-done)) * q_target

        q_val = self._model.q_critic.forward(obs, mask, act)
        loss_q = ((q_val - targets)**2).mean()

        return loss_q, targets
