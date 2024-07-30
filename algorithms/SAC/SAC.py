import itertools

import numpy as np
import torch
from torch.optim import Adam

from .kernel import RLActorCritic
from .replay_buffer import ReplayBuffer

from copy import deepcopy
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import EpochLogger, setup_logger_kwargs
from trajectory import RL4SysTrajectory

from algorithms._common.BaseAlgorithm import AlgorithmAbstract, count_vars

from conf_loader import ConfigLoader
"""Import and load RL4Sys/config.json SAC algorithm configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
config_loader = ConfigLoader(algorithm='SAC')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path


class SAC(AlgorithmAbstract):
    def __init__(self, kernel_size: int, kernel_dim: int, buf_size: int, act_dim: int = 1,
                 batch_size: int = hyperparams['batch_size'], seed: int = hyperparams['seed'],
                 traj_per_epoch: int = hyperparams['traj_per_epoch'], gamma: float = hyperparams['gamma'],
                 polyak: float = hyperparams['polyak'], alpha: float = hyperparams['alpha'],
                 lr: float = hyperparams['lr'], train_update_freq: int = hyperparams['train_update_freq'],
                 train_iters: int = hyperparams['train_iters']):

        super().__init__()
        seed += 10000 * os.getpid()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self._batch_size = batch_size
        self._traj_per_epoch = traj_per_epoch
        self._lr = lr
        self._gamma = gamma
        self._polyak = polyak
        self._alpha = alpha
        self._train_update_freq = train_update_freq
        self._train_iters = train_iters

        self._replay_buffer = ReplayBuffer(kernel_size * kernel_dim, kernel_size, buf_size, gamma)
        self._model = RLActorCritic(kernel_size * kernel_dim, act_dim)
        self._model_target = deepcopy(self._model)
        self._pi_optimizer = Adam(self._model.pi.parameters(), lr=lr)
        self._q_params = itertools.chain(self._model.q1.parameters(), self._model.q2.parameters())
        self._q_optimizer = Adam(self._q_params, lr=lr)

        current_dir = os.getcwd()
        log_data_dir = os.path.join(current_dir, './logs/')
        logger_kwargs = setup_logger_kwargs("rl4sys-sac-info", seed=seed, data_dir=log_data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.logger.setup_pytorch_saver(self._model)

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
            # Process each RL4SysAction in the trajectory
            ep_ret += r4a.rew
            ep_len += 1
            if not r4a.done:
                self._replay_buffer.store(r4a.obs, r4a.act, r4a.mask, r4a.rew, r4a.data['logp_a'])
                self.logger.store(LogPi=r4a.data['logp_a'])
            else:
                self._replay_buffer.finish_path(r4a.rew)
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)

        # get enough trajectories for training the model
        if self.traj > 0 and self.traj+1 % self._traj_per_epoch == 0:
            if self.traj % self._train_update_freq == 0:
                self.epoch += 1
                self.train_model()
                self.log_epoch()
                return True

        return False

    def train_model(self) -> None:
        """

        Returns:
        """
        data, batch = self._replay_buffer.get(self._batch_size)

        for i in range(self._train_iters):
            self._q_optimizer.zero_grad()
            loss_q, q_info = self.compute_loss_q(data)
            loss_q.backward()
            self._q_optimizer.step()

        self.logger.store(StopIter=i)

        for param in self._q_params:
            param.requires_grad = False

        for i in range(self._train_iters):
            self._pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self._pi_optimizer.step()

        for param in self._q_params:
            param.requires_grad = True

        with torch.no_grad():
            for param, param_target in zip(self._model.parameters(), self._model_target.parameters()):
                param_target.data.mul_(self._polyak)
                param_target.data.add_((1 - self._polyak) * param.data)

        q1_vals, q2_vals = q_info['Q1Vals'], q_info['Q2Vals']
        self.logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), Q1Vals=q1_vals, Q2Vals=q2_vals)

    def log_epoch(self) -> None:
        """Log the information collected in logger over the course of the last epoch
        """
        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('TestEpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TestEpLen', average_only=True)
        self.logger.log_tabular('Q1Vals', with_min_and_max=True)
        self.logger.log_tabular('Q2Vals', with_min_and_max=True)
        self.logger.log_tabular('LogPi', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('DeltaLossQ', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)
        self.logger.dump_tabular()

    def compute_loss_q(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        obs, act, rew, next_obs = data['obs'], data['act'], data['rew'], data['next_obs']

        q1 = self._model.q1.forward(obs, act)
        q2 = self._model.q2.forward(obs, act)

        with torch.no_grad():
            next_act, logp_next_act = self._model.pi(next_obs)

            q1_pi_target = self._model_target.q1.forward(next_obs, next_act)
            q2_pi_target = self._model_target.q2.forward(next_obs, next_act)
            q_pi_target = torch.min(q1_pi_target, q2_pi_target)
            q_pi_target = rew + self._gamma * (q_pi_target - self._alpha * logp_next_act)

        loss_q1 = ((q1 - q_pi_target)**2).mean()
        loss_q2 = ((q2 - q_pi_target)**2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        obs = data['obs']
        pi, logp_a = self._model.pi(obs)
        q1_pi = self._model.q1(obs, pi)
        q2_pi = self._model.q2(obs, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (self._alpha * logp_a - q_pi).mean()

        return loss_pi
