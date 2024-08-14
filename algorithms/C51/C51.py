from algorithms._common.BaseAlgorithm import AlgorithmAbstract

import numpy as np
import torch
from torch.optim import Adam

from .kernel import C51QNetwork
from .replay_buffer import ReplayBuffer

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import EpochLogger, setup_logger_kwargs
from trajectory import RL4SysTrajectory

from conf_loader import ConfigLoader

"""
Import and load RL4Sys/config.json DQN Agent configurations and applies them to
the current instance. 

Loads defaults if config.json is unavailable or key error thrown.
"""
config_loader = ConfigLoader(algorithm='C51')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path


"""
C51 Agent with hyperparameters
"""


class C51(AlgorithmAbstract):
    """
            Args:
                kernel_size: number of observations
                kernel_dim: number of features
                buf_size: size of replay buffer
                act_dim: number of actions (output dimension(s))
                batch_size: batch size of replay buffer
                seed: seed for random number generator
                traj_per_epoch: number of trajectories to be retrieved prior to training
                gamma: Q target discount factor
                epsilon: initial value for epsilon; exploration rate that is decayed over time
                epsilon_min: minimum value for epsilon
                epsilon_decay: decay rate for epsilon
                q_lr: learning rate for Q network, passed to Adam optimizer
                train_q_iters:
    """
    def __init__(self, kernel_size: int, kernel_dim: int, buf_size: int, act_dim: int = 1,
                 batch_size: int = hyperparams['batch_size'], seed: int = hyperparams['seed'],
                 traj_per_epoch: int = hyperparams['traj_per_epoch'], atoms: int = hyperparams['atoms'],
                 v_min: int = hyperparams['v_min'], v_max: int = hyperparams['v_max'],
                 gamma: float = hyperparams['gamma'], epsilon: float = hyperparams['epsilon'],
                 epsilon_min: float = hyperparams['epsilon_min'], epsilon_decay: float = hyperparams['epsilon_decay'],
                 q_lr: float = hyperparams['q_lr'], train_q_iters: int = hyperparams['train_q_iters']):

        super().__init__()
        seed += 10000 * os.getpid()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Hyperparameters
        self._batch_size = batch_size
        self._traj_per_epoch = traj_per_epoch
        self._atoms = atoms
        self._v_min = v_min
        self._v_max = v_max
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._train_q_iters = train_q_iters

        self._replay_buffer = ReplayBuffer(kernel_size * kernel_dim, kernel_size, buf_size, gamma=gamma, epsilon=epsilon)
        self._model = C51QNetwork(kernel_size, kernel_dim, act_dim, atoms, v_min, v_max, self._epsilon,
                                  self._epsilon_min, self._epsilon_decay)
        self._q_optimizer = Adam(self._model.parameters(), lr=q_lr)
        self._target_model = C51QNetwork(kernel_size, kernel_dim, act_dim, atoms, v_min, v_max, self._epsilon,
                                         self._epsilon_min, self._epsilon_decay)
        self._target_model.load_state_dict(self._model.state_dict())

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
                self._replay_buffer.store(r4a.obs, r4a.act, r4a.mask, r4a.rew, r4a.data['logp_a'])
                self.logger.store(QVals=r4a.data['logp_a'], Epsilon=r4a.data['epsilon'])
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

    def train_model(self) -> None:
        """Train model on data from DQN replay_buffer.
        """
        data, batch = self._replay_buffer.get(self._batch_size)

        q_l_old, pmf_target_old = self.compute_loss_q(data)
        q_l_old = q_l_old.item()
        pmf_target_old = pmf_target_old.item()

        # Train Q network for n iterations of gradient descent
        for i in range(self._train_q_iters):
            self._q_optimizer.zero_grad()
            loss_q, pmf_target = self.compute_loss_q(data)
            loss_q.backward()
            self._q_optimizer.step()

        self.logger.store(StopIter=i)

        self.logger.store(PMFTargets=pmf_target.item(), LossQ=loss_q.item(), DeltaLossQ=abs(loss_q.item() - q_l_old),
                          DeltaPMFTargets=abs(pmf_target.item() - pmf_target_old))

    def log_epoch(self) -> None:
        """Log the information collected in logger over the course of the last epoch
        """
        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('Epsilon', with_min_and_max=True)
        self.logger.log_tabular('QVals', average_only=True)
        self.logger.log_tabular('QTargets', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('DeltaLossQ', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)
        self.logger.dump_tabular()

    def compute_loss_q(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss for Q function.

        Args:
            data: dictionary containing batched data from replay buffer
        Returns:
            Loss for Q function, Q target for logging

        """
        obs, mask, rew, next_obs = data['obs'], data['mask'], data['rew'], data['next_obs']

        # Projection and Q-Loss
        with torch.no_grad():
            next_pmf = self._target_model.step(next_obs, mask)[1]['q_pmf'].logits
            next_atoms = rew.unsqueeze(-1) + self._gamma * self._target_model.atoms

            delta_z = self._target_model.atoms[1] - self._target_model.atoms[0]
            target_z = next_atoms.clamp(self._v_min, self._v_max)

            b = (target_z - self._v_min) / delta_z
            l = b.floor().clamp(0, self._atoms - 1).long()
            u = b.ceil().clamp(0, self._atoms - 1).long()

            d_m_l = ((u + (l == u).float() - b) * next_pmf).to(next_pmf.dtype)
            d_m_u = ((b - l) * next_pmf).to(next_pmf.dtype)
            target_pmf = torch.zeros_like(next_pmf)
            for i in range(target_pmf.size(0)):
                target_pmf[i].index_add_(0, l[i], d_m_l[i])
                target_pmf[i].index_add_(0, u[i], d_m_u[i])

        # loss_q probability mass function
        old_pmf = self._model.step(obs, mask)[1]['q_pmf'].logits
        loss_q = (-(target_pmf * old_pmf.clamp(min=1e-5, max=1-1e5).log()).sum(-1)).mean()

        return loss_q, target_pmf
