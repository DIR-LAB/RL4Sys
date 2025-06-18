# Standard library imports
import os
import threading
import time
import random
import numpy as np

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# Local imports
from rl4sys.algorithms.DQN.kernel import DeepQNetwork
from rl4sys.algorithms.DQN.replay_buffer import ReplayBuffer
from rl4sys.common.trajectory import RL4SysTrajectory

class DQN():
    """
    Example DQN agent *modified* to accept entire trajectories at once (like your PPO snippet).
    The agent will only train after collecting `_traj_per_epoch` trajectories, rather than
    training continuously every few steps.
    """

    def __init__(self,  
                 version: int,
                 seed: int = 0,
                 input_size: int = 1,
                 act_dim: int = 1,
                 buf_size: int = 1000000,
                 batch_size: int = 32,
                 gamma: float = 0.99,
                 lr: float = 1e-4,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 target_update: int = 1000,
                 learning_rate: float = 1e-4,
                 tau: float = 0.005,
                 target_network_frequency: int = 1000,
                 buffer_size: int = 1000000,
                 traj_per_epoch: int = 5,
                 start_e: float = 1.0,
                 end_e: float = 0.05,
                 exploration_fraction: float = 0.1,
                 learning_starts: int = 50000,
                 train_frequency: int = 4,
                 total_timesteps: int = 1000000,
                 train_iters: int = 1,
                 ):
        
        # Hyperparameters
        self._buf_size = buf_size
        self._batch_size = batch_size
        self._gamma = gamma
        self._lr = lr
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._target_update = target_update
        self.version = version
        self.seed = seed
        self.type = "offpolicy"
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create Q-network
        self.q_network = DeepQNetwork(input_size, act_dim).to(self.device)
        self.models = {}
        # Initialize lock for thread-safe model updates
        self.lock = threading.RLock()
        self.models[self.version] = self.q_network


        self.target_network = DeepQNetwork(input_size, act_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Create optimizer
        self.optimizer = Adam(self.q_network.parameters(), lr=self._lr)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=input_size,
            buf_size=buffer_size,
            gamma=gamma,
            epsilon=start_e
        )
        
        # Save references
        self.tau = tau
        self.target_network_frequency = target_network_frequency
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.total_timesteps = total_timesteps
        self.train_iters = train_iters
        self._traj_per_epoch = traj_per_epoch

        # Initialize metrics
        self.ep_rewards = 0
        self.start_time = None
        self.steps = 0
        self.epoch = 0
        self.global_step = 0

        # Set up logger
        log_data_dir = os.path.join('./logs/rl4sys-dqn-info', f"{int(time.time())}__{self.seed}")
        os.makedirs(log_data_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_data_dir)

    def save(self, filename: str) -> None:
        new_path = os.path.join(self.save_model_path, filename + ('.pth' if not filename.endswith('.pth') else ''))
        torch.save(self.q_network, new_path)

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
            self._traj_per_epoch += 1
            self.global_step += 1

            self.ep_rewards += r4a.rew
            # Store in replay buffer
            # Our PPO snippet simply stored transitions in a list, but for DQN,
            # we store them into replay_buffer.
            self.replay_buffer.store(r4a.obs, r4a.act, r4a.rew, r4a.data['q_val'], r4a.done)

            if r4a.done:
                self.writer.add_scalar("charts/reward", self.ep_rewards, self.global_step)
                self.ep_rewards = 0

            if self._traj_per_epoch >= self._batch_size:
                if self.global_step > self.learning_starts:
                    self.epoch += 1
                    loss_q, q_vals = self.train_model()
                    update = True

                    self.writer.add_scalar("losses/td_loss", loss_q, self.global_step)
                    self.writer.add_scalar("losses/q_values", q_vals, self.global_step)
                    # print("SPS:", int(self.global_step / (time.time() - self.start_time)))
                    self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
                    
        # Once we have enough trajectories, do an update
        
        return update

    def get_current_model(self):
        with self.lock:
            return self.models[self.version], self.version
        
    def train_model(self) -> None:
        """
        Train the model after a number of complete trajectories have been collected.
        You can choose how many minibatch updates to do per 'train_model()' call.
        Below we do a certain number of gradient steps, or you can do 1 step per call.
        """
        for _ in range(self.train_iters):
            data, _ = self.replay_buffer.get(self._batch_size)
            obs      = data['obs']
            next_obs = data['next_obs']
            act      = data['act'].long()
            rew      = data['rew']
            done     = data['done']

            with torch.no_grad():
                q_next = self.target_network.forward(next_obs)
                q_next_max, _ = q_next.max(dim=1)
                td_target = rew + self._gamma * q_next_max * (1 - done)

            q_vals = self.q_network.forward(obs)  # shape [batch_size, act_dim]
            q_taken = q_vals.gather(1, act.unsqueeze(-1)).squeeze(-1)

            loss_q = F.mse_loss(q_taken, td_target)

            self.optimizer.zero_grad()
            loss_q.backward()
            self.optimizer.step()

            # Possibly update target net
            if self.epoch % self.target_network_frequency == 0:
                self._update_target()

                # finish training, update model
        with self.lock:
            self.version += 1
            self.models[self.version] = self.q_network

        return loss_q, q_vals.mean().item()

    def _update_target(self):
        """
        Hard or soft update for the target network.
        """
        for param, target_param in zip(self.q_network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )


