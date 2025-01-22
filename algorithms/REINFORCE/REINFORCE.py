from _common._algorithms.BaseAlgorithm import AlgorithmAbstract

import numpy as np
import torch
from torch.optim import Adam

from .kernel import PolicyWithoutBaseline, PolicyWithBaseline
from .replay_buffer import ReplayBuffer

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.logger import EpochLogger, setup_logger_kwargs
from trajectory import RL4SysTrajectory

from conf_loader import ConfigLoader
"""Import and load RL4Sys/config.json REINFORCE algorithm configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
config_loader = ConfigLoader(algorithm='REINFORCE')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path


class REINFORCE(AlgorithmAbstract):
    def __init__(self, env_dir: str, kernel_size: int, kernel_dim: int, act_dim: int, buf_size: int,
                 discrete: bool = hyperparams['discrete'], with_vf_baseline: bool = hyperparams['with_vf_baseline'], seed: int = hyperparams['seed'],
                 traj_per_epoch: int = hyperparams['traj_per_epoch'], gamma: float = hyperparams['gamma'],
                 lam: float = hyperparams['lam'], pi_lr: float = hyperparams['pi_lr'],
                 vf_lr: float = hyperparams['vf_lr'], train_vf_iters: int = hyperparams['train_vf_iters']):

        super().__init__()
        seed += 10000 * os.getpid()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Hyperparameters
        self._with_vf_baseline = with_vf_baseline
        self._traj_per_epoch = traj_per_epoch
        self._gamma = gamma
        self._lam = lam
        self._pi_lr = pi_lr
        self._vf_lr = vf_lr
        self._train_vf_iters = train_vf_iters

        self._replay_buffer = ReplayBuffer(kernel_size * kernel_dim, act_dim, buf_size, gamma, lam, with_vf_baseline)
        if with_vf_baseline:
            self._model = PolicyWithBaseline(kernel_size * kernel_dim, act_dim, discrete, [128, 128], torch.nn.ReLU)
            self._vf_optimizer = Adam(self._model.baseline.parameters(), lr=vf_lr)
        else:
            self._model = PolicyWithoutBaseline(kernel_size * kernel_dim, act_dim, discrete, [128, 128])
        self._pi_optimizer = Adam(self._model.parameters(), lr=pi_lr)

        # set up logger
        log_data_dir = os.path.join(env_dir, './logs/')
        logger_kwargs = setup_logger_kwargs(
            "rl4sys-reinforce-info", seed=seed, data_dir=log_data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        self.traj = 0
        self.epoch = 0

    def save(self, filename: str) -> None:
        new_path = os.path.join(save_model_path, filename + ('.pth' if not filename.__contains__('.pth') else ''))
        torch.save(self._model, new_path)

    def receive_trajectory(self, trajectory: RL4SysTrajectory) -> bool:
        self.traj += 1
        ep_ret, ep_len = 0, 0

        for r4a in trajectory.actions:
            ep_ret += r4a.rew
            ep_len += 1
            if not r4a.done:
                if self._with_vf_baseline:
                    self._replay_buffer.store(r4a.obs, r4a.act, r4a.mask, r4a.rew, r4a.data['v'], r4a.data['logp_a'])
                    self.logger.store(VVals=r4a.data['v'])
                else:
                    self._replay_buffer.store(r4a.obs, r4a.act, r4a.mask, r4a.rew, None, r4a.data['logp_a'])
            else:
                self._replay_buffer.finish_path(r4a.rew)
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)

        if self.traj > 0 and self.traj % self._traj_per_epoch == 0:
            self.epoch += 1
            self.train_model()
            self.log_epoch()
            return True

        return False

    def train_model(self) -> None:
        data = self._replay_buffer.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        if self._with_vf_baseline:
            v_l_old = self.compute_loss_vf(data).item()

        self._pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self._pi_optimizer.step()

        if self._with_vf_baseline:
            for i in range(self._train_vf_iters):
                self._vf_optimizer.zero_grad()
                loss_v = self.compute_loss_vf(data)
                loss_v.backward()
                self._vf_optimizer.step()

        kl, ent = pi_info['kl'], pi_info['ent']
        delta_loss_pi = loss_pi.item() - pi_l_old
        delta_loss_v = 0 if not self._with_vf_baseline else loss_v.item() - v_l_old

        self.logger.store(LossPi=loss_pi.item(), DeltaLossPi=delta_loss_pi,
                          KL=kl, Entropy=ent)

        if self._with_vf_baseline:
            self.logger.store(LossV=loss_v.item(), DeltaLossV=delta_loss_v)

    def log_epoch(self) -> None:
        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('DeltaLossPi', average_only=True)
        if self._with_vf_baseline:
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
        self.logger.dump_tabular()

    def compute_loss_pi(self, data):
        obs, mask, act, adv, logp_old = data['obs'], data['mask'], data['act'], data['adv'], data['logp']

        pi, logp = self._model.policy.forward(obs, mask, act)
        loss_pi = -(logp * adv).mean()

        approx_kl = (logp_old - logp).mean().item()
        entropy = pi.entropy().mean().item()

        pi_info = dict(kl=approx_kl, ent=entropy)

        return loss_pi, pi_info

    def compute_loss_vf(self, data):
        obs, mask, ret = data['obs'], data['mask'], data['ret']
        return ((self._model.baseline.forward(obs, mask) - ret) ** 2).mean()
