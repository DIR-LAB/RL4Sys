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
from torch.utils.tensorboard import SummaryWriter

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
        train_iters: int = hyperparams['train_iters'],
    ):
        super().__init__()

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Logger setup
        log_data_dir = os.path.join(env_dir, './logs/rl4sys-dqn-info')
        log_data_dir = os.path.join(log_data_dir, f"{int(time.time())}__{seed}")
        os.makedirs(log_data_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_data_dir)

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
        self.train_iters = train_iters
        # New in this version:
        self._traj_per_epoch = traj_per_epoch


        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.traj = 0
        self.start_time = None
        self.ep_rewards = 0



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

        for i, r4a in enumerate(trajectory):
            self.traj += 1
            self.global_step += 1

            self.ep_rewards += r4a.rew
            # Store in replay buffer
            # Our PPO snippet simply stored transitions in a list, but for DQN,
            # we store them into replay_buffer.
            self.replay_buffer.store(r4a.obs, r4a.act, r4a.mask, r4a.rew, r4a.data['q_val'], r4a.done)

            if r4a.done:
                self.writer.add_scalar("charts/reward", self.ep_rewards, self.global_step)
                self.ep_rewards = 0

            if self.traj >= self.batch_size:
                if self.global_step > self.learning_starts:
                    self.epoch += 1
                    loss_q, q_vals = self.train_model()
                    #self.log_epoch()
                    update = True

                    self.writer.add_scalar("losses/td_loss", loss_q, self.global_step)
                    self.writer.add_scalar("losses/q_values", q_vals, self.global_step)
                    # print("SPS:", int(self.global_step / (time.time() - self.start_time)))
                    self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
                    
        
            
        # Once we have enough trajectories, do an update
        
        return update

    def train_model(self) -> None:
        """
        Train the model after a number of complete trajectories have been collected.
        You can choose how many minibatch updates to do per 'train_model()' call.
        Below we do a certain number of gradient steps, or you can do 1 step per call.
        """
        for _ in range(self.train_iters):
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
            
        return loss_q, q_vals.mean().item()



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
