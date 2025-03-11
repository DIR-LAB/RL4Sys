from _common._algorithms.BaseAlgorithm import AlgorithmAbstract

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import time, os

from .kernel import DeepQNetwork
from .replay_buffer import ReplayBuffer
from utils.logger import EpochLogger, setup_logger_kwargs
from utils.conf_loader import ConfigLoader
from protocol.action import RL4SysAction
from protocol.trajectory import RL4SysTrajectory


############
#  CONFIG  #
############
config_loader = ConfigLoader(algorithm='DQN')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path

class DQN(AlgorithmAbstract):
    """
    Example DQN agent *modified* to accept entire trajectories at once (like your PPO snippet).
    The agent will only train after collecting `_traj_per_epoch` trajectories, rather than
    training continuously every few steps.
    """

    def __init__(
        self,
        env_dir: str,
        input_size: int,
        act_dim: int,
        buf_size: int,
        ############### Shared CleanRL-like hyperparams ###############
        seed: int = hyperparams['seed'],
        learning_rate: float = hyperparams['learning_rate'],
        batch_size: int = hyperparams['batch_size'],
        gamma: float = hyperparams['gamma'],
        tau: float = hyperparams['tau'],
        target_network_frequency: int = hyperparams['target_network_frequency'],
        buffer_size: int = hyperparams['buffer_size'],
        ############### Some extra hyperparams #########################
        # We'll add one that parallels PPO's "traj_per_epoch":
        traj_per_epoch: int = 5,
        ###############################################################
        start_e: float = hyperparams['start_e'],
        end_e: float = hyperparams['end_e'],
        exploration_fraction: float = hyperparams['exploration_fraction'],
        learning_starts: int = hyperparams['learning_starts'],
        train_frequency: int = hyperparams['train_frequency'],
        total_timesteps: int = hyperparams['total_timesteps'],
    ):
        super().__init__()

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Log setup
        log_data_dir = os.path.join(env_dir, './logs/')
        logger_kwargs = setup_logger_kwargs("rl4sys-dqn-info", seed=seed, data_dir=log_data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        # DQN networks
        self._model = DeepQNetwork(input_size, act_dim)
        self.q_target = DeepQNetwork(input_size, act_dim)
        self.q_target.load_state_dict(self._model.state_dict())

        self.optimizer = Adam(self._model.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=input_size,
            mask_dim=act_dim,
            buf_size=buffer_size,
            gamma=gamma,
            epsilon=start_e
        )

        # Save references
        self.gamma = gamma
        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self.batch_size = batch_size
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.total_timesteps = total_timesteps
        # New in this version:
        self._traj_per_epoch = traj_per_epoch

        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.traj = 0
        self.start_time = None

        self.logger.setup_pytorch_saver(self._model)

    def save(self, filename: str) -> None:
        new_path = os.path.join(save_model_path, filename + ('.pth' if not filename.endswith('.pth') else ''))
        torch.save(self._model, new_path)

    def receive_trajectory(self, trajectory: RL4SysTrajectory) -> bool:
        """
        Process a trajectory from the environment (similar to PPO's approach).

        - We'll store each step (RLA) in the replay buffer.
        - Log episode returns & lengths when `done=True`.
        - Once we've collected `self._traj_per_epoch` total trajectories,
          we call `train_model()` and `log_epoch()`.
        - Return True iff we just finished an epoch (meaning an update occurred).
        """
        if self.start_time is None:
            self.start_time = time.time()

        update = False
        self.traj += 1
        ep_ret, ep_len = 0.0, 0

        for i, r4a in enumerate(trajectory):
            self.global_step += 1
            ep_ret += r4a.rew
            ep_len += 1

            # Store in replay buffer
            # Our PPO snippet simply stored transitions in a list, but for DQN,
            # we store them into replay_buffer.
            self.replay_buffer.store(r4a.obs, r4a.act, r4a.mask, r4a.rew, r4a.data['q_val'], r4a.done)
            self.logger.store(QVals=r4a.data['q_val'], Epsilon=r4a.data['epsilon'])

            if r4a.done:
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                ep_ret, ep_len = 0.0, 0

        if self.traj > self.learning_starts:
            self.epoch += 1
            self.train_model()
            self.log_epoch()
            update = True

                
        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
        ep_ret, ep_len = 0.0, 0
                

        # Once we have enough trajectories, do an update
        
        return update

    def train_model(self) -> None:
        """
        Train the model after a number of complete trajectories have been collected.
        You can choose how many minibatch updates to do per 'train_model()' call.
        Below we do a certain number of gradient steps, or you can do 1 step per call.
        """
        # Example: let's do 10 gradient steps every epoch
        num_train_steps = 10

        for _ in range(num_train_steps):
            data, _ = self.replay_buffer.get(self.batch_size)
            obs      = data['obs']
            next_obs = data['next_obs']
            act      = data['act'].long()
            rew      = data['rew']
            done     = data['done']
            mask     = data['mask']

            with torch.no_grad():
                q_next = self.q_target.forward(next_obs, mask)
                q_next_max, _ = q_next.max(dim=1)
                td_target = rew + self.gamma * q_next_max * (1 - done)

            q_vals = self._model.forward(obs, mask)  # shape [batch_size, act_dim]
            q_taken = q_vals.gather(1, act.unsqueeze(-1)).squeeze(-1)

            loss_q = F.mse_loss(q_taken, td_target)

            self.optimizer.zero_grad()
            loss_q.backward()
            self.optimizer.step()

            # Possibly update target net
            if self.epoch % self.target_network_frequency == 0:
                self._update_target()

            # Optionally log
            self.logger.store(LossQ=loss_q.item())

    def _update_target(self):
        """
        Hard or soft update for the target network.
        """
        for param, target_param in zip(self._model.parameters(), self.q_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def log_epoch(self) -> None:
        """
        Similar to PPO's log_epoch(). Log interesting stats, then dump tabular.
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        sps = int(self.global_step / elapsed) if elapsed > 0 else 0

        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', with_min_and_max=True)
        #self.logger.log_tabular('LossQ', average_only=True)
        #self.logger.log_tabular('SPS', sps)
        self.logger.dump_tabular()
