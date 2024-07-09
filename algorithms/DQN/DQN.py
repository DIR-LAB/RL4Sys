import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .kernel import DeepQNetwork
from .replay_buffer import ReplayBuffer

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import EpochLogger, setup_logger_kwargs
from trajectory import RL4SysTrajectory

import json
"""
Import and load RL4Sys/config.json DQN Agent configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
CONFIG_PATH = os.path.join(top_dir, 'config.json')
hyperparams = {}
save_model_path = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        hyperparams = config['algorithms']
        hyperparams = hyperparams['DQN']
        save_model_path = config['model_paths']
        save_model_path = os.path.join(save_model_path['save_model'])
except (FileNotFoundError, KeyError):
    print(f"DQN: Failed to load configuration from {CONFIG_PATH}, loading defaults.")
    hyperparams = {
        "batch_size": 32,
        "seed": 0,
        "traj_per_epoch": 3,
        "clip_value": 100,
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 5e-4,
        "train_update_freq": 0.005,
        "q_lr": 1e-3,
        "train_q_iters": 80
    }
    save_model_path = os.path.join(top_dir, 'models/model.pth')


"""
DQN Agent with hyperparameters
"""


class DQN:
    """
            Args:
                kernel_size:
                kernel_dim:
                buf_size:
                act_size:
                batch_size:
                seed:
                traj_per_epoch:
                gamma:
                epsilon:
                epsilon_min:
                epsilon_decay:
                train_update_freq:
                q_lr:
                train_q_iters:
    """
    def __init__(self, kernel_size: int, kernel_dim: int,
                 buf_size: int, act_dim: int = 1, batch_size: int = hyperparams['batch_size'],
                 seed: int = hyperparams['seed'],
                 traj_per_epoch: int = hyperparams['traj_per_epoch'],
                 clip_value: float = hyperparams['clip_value'],
                 gamma: float = hyperparams['gamma'],
                 epsilon: float = hyperparams['epsilon'],
                 epsilon_min: float = hyperparams['epsilon_min'],
                 epsilon_decay: float = hyperparams['epsilon_decay'],
                 train_update_freq: float = hyperparams['train_update_freq'],
                 q_lr: float = hyperparams['q_lr'],
                 train_q_iters: int = hyperparams['train_q_iters']):

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
        self._clip_value = clip_value
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._train_update_freq = train_update_freq
        self._train_q_iters = train_q_iters

        self._action_space = [i for i in range(act_dim)]
        self._replay_buffer = ReplayBuffer(kernel_size * kernel_dim, kernel_size, buf_size, gamma=gamma, epsilon=epsilon)
        self._model = DeepQNetwork(kernel_size, kernel_dim, act_dim)
        self._q_optimizer = Adam(self._model.parameters(), lr=q_lr)

        # set up logger
        current_dir = os.getcwd()
        log_data_dir = os.path.join(current_dir, './logs/')
        logger_kwargs = setup_logger_kwargs(
            "rl4sys-dqn-scheduler", seed=seed, data_dir=log_data_dir)
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
                self._replay_buffer.store(r4a.obs, r4a.act, r4a.mask, r4a.rew, r4a.data['q_val'], r4a.data['next_obs'])
                self.logger.store(QVals=r4a.data['q_val'])
            else:
                self._replay_buffer.finish_path(r4a.rew)
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)

        # get enough trajectories for training the model
        if self.traj > 0 and self.traj % self._traj_per_epoch == 0:
            self.epoch += 1
            self.train_model()
            self.log_epoch()
            return True

        return False

    def act(self, obs: torch.Tensor, mask: torch.Tensor) -> 'np.ndarray':
        """
        
        """
        if np.random.random() >= self._epsilon:
            return self._model.step(obs, mask)[0]
        else:
            return np.random.choice(self._action_space)

    def train_model(self) -> None:
        """Train model on data from DQN replay_buffer.
        """
        data, batch = self._replay_buffer.get(self._batch_size)

        # Train Q network for n iterations of gradient descent
        for i in range(self._train_q_iters):
            self._q_optimizer.zero_grad()
            loss_q, q_info = self.compute_loss_q(data, batch)
            loss_q.backward()
            self._q_optimizer.step()

            q_val, q_target = q_info['q_val'], q_info['q_target']

            self.logger.store(QVals=q_val, QTargets=q_target, LossQ=loss_q,
                              DeltaLossQ=abs(loss_q - (self.logger.get_stats('LossQ')[-1]
                                                       if self.logger.get_stats('LossQ')[-1] else 0)))

        self._epsilon = max(self._epsilon - self._epsilon_decay, self._epsilon_min)

    def log_epoch(self) -> None:
        """Log the information collected in logger over the course of the last epoch
        """
        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('Epsilon', with_min_and_max=True)
        self.logger.log_tabular('QVals', with_min_and_max=True)
        self.logger.log_tabular('QTargets', with_min_and_max=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('DeltaLossQ', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)
        self.logger.dump_tabular()

    def compute_loss_q(self, data: dict[str, torch.Tensor], batch) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss for Q function.

        Args:
            data: dictionary containing all data from replay buffer
            batch: batch indices for replay buffer
        Returns:
            Loss for Q function, statistics for logging

        """
        # TODO: validate data input element contents for given environment/example context
        obs, act, mask, rew, next_obs = data['obs'], data['act'], data['mask'], data['rew'], data['next_obs']

        batch_idx = np.arange(self._batch_size, dtype=np.int32)
        act_batch = act.numpy()[batch]

        # Q loss
        q_val = self._model.forward(obs, mask, softmax=True)[batch_idx, act_batch]
        next_q_val = self._model.forward(next_obs, mask, softmax=True)
        q_target = rew + self._gamma * torch.max(next_q_val.detach(), dim=1)[0]
        # Mean Square Error (MSE) loss
        loss_q = ((q_val - q_target.detach())**2).mean()

        # Q information
        q_info = dict(q_val=q_val, q_target=q_target)

        return loss_q, q_info
