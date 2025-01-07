from _common._algorithms.BaseAlgorithm import AlgorithmAbstract

import itertools

import numpy as np
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from .kernel import DoubleQActorCritic, DoubleQFunction
from .replay_buffer import ReplayBuffer

from copy import deepcopy
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import EpochLogger, setup_logger_kwargs
from trajectory import RL4SysTrajectory

from conf_loader import ConfigLoader
"""Import and load RL4Sys/config.json SAC algorithm configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
config_loader = ConfigLoader(algorithm='SAC')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path


class SAC(AlgorithmAbstract):
    def __init__(self, env_dir: str, kernel_size: int, kernel_dim: int, buf_size: int, act_dim: int = hyperparams['act_dim'],
                 discrete: bool = hyperparams['discrete'], adaptive_alpha: bool = hyperparams['adaptive_alpha'],
                 batch_size: int = hyperparams['batch_size'], seed: int = hyperparams['seed'],
                 traj_per_epoch: int = hyperparams['traj_per_epoch'], log_std_min: int = hyperparams['log_std_min'],
                 log_std_max: int = hyperparams['log_std_max'], gamma: float = hyperparams['gamma'],
                 polyak: float = hyperparams['polyak'], alpha: float = hyperparams['alpha'],
                 lr: float = hyperparams['lr'], clip_grad_name: float = hyperparams['clip_grad_norm'],
                 train_update_freq: int = hyperparams['train_update_freq'], train_iters: int = hyperparams['train_iters']):

        super().__init__()
        seed += 10000 * os.getpid()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # hyperparameters
        self._discrete = discrete
        self._adaptive_alpha = adaptive_alpha
        self._batch_size = batch_size
        self._traj_per_epoch = traj_per_epoch
        self._lr = lr
        self._gamma = gamma
        self._polyak = polyak
        self._alpha = alpha
        self._clip_grad_norm = clip_grad_name
        self._train_update_freq = train_update_freq
        self._train_iters = train_iters

        self._replay_buffer = ReplayBuffer(kernel_size * kernel_dim, act_dim, buf_size, gamma)
        self._model = DoubleQActorCritic(kernel_size * kernel_dim, kernel_dim, (256, 256),
                                         torch.nn.ReLU, log_std_min, log_std_max, discrete, seed)

        self._target_critic = DoubleQFunction(kernel_size * kernel_dim, (256, 256), kernel_dim,
                                              torch.nn.ReLU, discrete, seed)
        self._target_critic.q1.load_state_dict(self._model.q.q1.state_dict())
        self._target_critic.q2.load_state_dict(self._model.q.q2.state_dict())
        for param in self._target_critic.parameters():
            param.requires_grad = False

        self._pi_optimizer = Adam(self._model.pi.parameters(), lr=lr)
        self._q1_optimizer = Adam(self._model.q.q1.parameters(), lr=lr)
        self._q2_optimizer = Adam(self._model.q.q2.parameters(), lr=lr)

        if self._adaptive_alpha:
            self._target_entropy = -act_dim
            self._log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True)
            self._alpha = self._log_alpha.exp().detach()
            self._alpha_optimizer = Adam([self._log_alpha], lr=lr)

        log_data_dir = os.path.join(env_dir, './logs/')
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
                self._replay_buffer.store(r4a.obs, r4a.next_obs, r4a.act, r4a.mask, r4a.rew)
            else:
                self._replay_buffer.finish_path(r4a.rew)
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)

        # get enough trajectories for training the model
        if self.traj > 0 and self.traj % self._traj_per_epoch == 0:
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

        q1_l_old, q2_l_old, _ = self.compute_loss_critic(data)
        q_l_old = q1_l_old + q2_l_old
        pi_l_old, log_act_probs_old = self.compute_loss_actor(data)
        if self._adaptive_alpha:
            alpha_l_old = self.compute_loss_entropy(log_act_probs_old)

        # train critic
        for i in range(self._train_iters):
            loss_q1, loss_q2, q_info = self.compute_loss_critic(data)
            # q1 loss propagation
            self._q1_optimizer.zero_grad()
            loss_q1.backward(retain_graph=True)
            clip_grad_norm_(self._model.q.q1.parameters(), self._clip_grad_norm)
            self._q1_optimizer.step()
            # q2 loss propagation
            self._q2_optimizer.zero_grad()
            loss_q2.backward()
            clip_grad_norm_(self._model.q.q2.parameters(), self._clip_grad_norm)
            self._q2_optimizer.step()

        self.logger.store(StopIter=i)

        for param_q1 in self._model.q.q1.parameters():
            param_q1.requires_grad = False
        for param_q2 in self._model.q.q2.parameters():
            param_q2.requires_grad = False

        # train actor
        for i in range(self._train_iters):
            self._pi_optimizer.zero_grad()
            loss_pi, log_act_probs = self.compute_loss_actor(data)
            loss_pi = loss_pi.requires_grad_()
            loss_pi.backward()
            self._pi_optimizer.step()

        for param_q1 in self._model.q.q1.parameters():
            param_q1.requires_grad = True
        for param_q2 in self._model.q.q2.parameters():
            param_q2.requires_grad = True

        # train adaptive alpha if enabled
        if self._adaptive_alpha:
            for i in range(self._train_iters):
                self._alpha_optimizer.zero_grad()
                loss_alpha = self.compute_loss_entropy(log_act_probs)
                loss_alpha.backward()
                self._alpha_optimizer.step()
                self._alpha = self._log_alpha.exp().detach()

            self.logger.store(Alpha=self._alpha, LossAlpha=loss_alpha.item(),
                              DeltaLossAlpha=abs(loss_alpha.item() - alpha_l_old.item()))

        for param, param_target in zip(self._model.q.q1.parameters(), self._target_critic.q1.parameters()):
            param_target.data.copy_(self._polyak * param.data + (1 - self._polyak) * param_target.data)
        for param, param_target in zip(self._model.q.q2.parameters(), self._target_critic.q2.parameters()):
            param_target.data.copy_(self._polyak * param.data + (1 - self._polyak) * param_target.data)

        q1_vals, q2_vals = q_info['Q1Vals'], q_info['Q2Vals']
        loss_q = loss_q1 + loss_q2
        self.logger.store(Q1Vals=q1_vals, Q2Vals=q2_vals, LossQ=loss_q.item(), LossPi=loss_pi.item(),
                          DeltaLossQ=abs(loss_q.item() - q_l_old.item()), DeltaLossPi=abs(loss_pi.item() - pi_l_old.item()))

    def log_epoch(self) -> None:
        """Log the information collected in logger over the course of the last epoch
        """
        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        if self._adaptive_alpha:
            self.logger.log_tabular('Alpha', average_only=True)
        self.logger.log_tabular('Q1Vals', average_only=True)
        self.logger.log_tabular('Q2Vals', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        if self._adaptive_alpha:
            self.logger.log_tabular('LossAlpha', average_only=True)
        self.logger.log_tabular('DeltaLossQ', average_only=True)
        self.logger.log_tabular('DeltaLossPi', average_only=True)
        if self._adaptive_alpha:
            self.logger.log_tabular('DeltaLossAlpha', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)
        self.logger.dump_tabular()

    def compute_loss_critic(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        obs, mask, act, rew, next_obs = data['obs'], data['mask'], data['act'], data['rew'], data['next_obs']

        with torch.no_grad():
            _, next_probs, logp_a = self._model.pi(obs, mask)

            target_q1, target_q2 = self._target_critic.forward(next_obs, mask)
            target_q = next_probs * (torch.min(target_q1, target_q2) - self._alpha * logp_a)
            target_q = rew + (self._gamma * target_q.sum(1).unsqueeze(-1))

        curr_q1, curr_q2 = self._model.q(obs, mask)
        act = act.view(-1, 1)
        curr_q1 = curr_q1.gather(1, act.long())
        curr_q2 = curr_q2.gather(1, act.long())

        loss_q1 = ((curr_q1 - target_q) ** 2).mean() * 0.5
        loss_q2 = ((curr_q2 - target_q) ** 2).mean() * 0.5

        q_info = dict(Q1Vals=curr_q1.detach().numpy(),
                      Q2Vals=curr_q2.detach().numpy())

        return loss_q1, loss_q2, q_info

    def compute_loss_actor(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        obs, mask = data['obs'], data['mask']

        with torch.no_grad():
            _, probs, logp_a = self._model.pi(obs, mask)
            q1, q2 = self._model.q(obs, mask)
            min_q = torch.min(q1, q2)

        loss_pi = torch.mean((probs * (self._alpha * logp_a - min_q)).sum(1))
        log_act_probs = torch.sum(logp_a * probs, dim=1)

        return loss_pi, log_act_probs

    def compute_loss_entropy(self, log_act_probs: torch.Tensor) -> torch.Tensor:
        loss_alpha = -torch.mean(self._log_alpha.exp() * (log_act_probs + self._target_entropy).detach())

        return loss_alpha
