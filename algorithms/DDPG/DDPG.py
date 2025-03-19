from _common._algorithms.BaseAlgorithm import AlgorithmAbstract
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import time
import os
from utils.logger import EpochLogger
from utils.conf_loader import ConfigLoader
from protocol.trajectory import RL4SysTrajectory
from torch.utils.tensorboard import SummaryWriter

from .kernel import Actor, Critic
from .replay_buffer import ReplayBuffer

############
#  CONFIG  #
############
config_loader = ConfigLoader(algorithm='DDPG')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path

class DDPG(AlgorithmAbstract):
    def __init__(
        self,
        env_dir: str,
        input_size: int,
        act_dim: int,
        act_limit: float,
        seed: int = hyperparams['seed'],
        gamma: float = hyperparams['gamma'],
        tau: float = hyperparams['tau'],
        learning_rate: float = hyperparams['learning_rate'],
        batch_size: int = hyperparams['batch_size'],
        buffer_size: int = hyperparams['buffer_size'],
        exploration_noise: float = hyperparams['exploration_noise'],
        learning_starts: int = hyperparams['learning_starts'],
        policy_frequency: int = hyperparams['policy_frequency'],
        traj_per_epoch: int = 5,
    ):
        super().__init__()
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create actor-critic module
        self.actor = Actor(input_size, act_dim, act_limit)
        self.critic = Critic(input_size, act_dim)
        
        # Create target networks
        self.actor_target = Actor(input_size, act_dim, act_limit)
        self.critic_target = Critic(input_size, act_dim)
        
        # Copy target parameters
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Set up optimizers
        self.pi_optimizer = Adam(self.actor.parameters(), lr=learning_rate)
        self.q_optimizer = Adam(self.critic.parameters(), lr=learning_rate)

        # Set up replay buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=input_size,
            act_dim=act_dim,
            buf_size=buffer_size,
            gamma=gamma
        )

        # Logging
        log_data_dir = os.path.join(env_dir, './logs/rl4sys-ddpg-info')
        log_data_dir = os.path.join(log_data_dir, f"{seed}__{int(time.time())}")
        os.makedirs(log_data_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_data_dir)

        # Save hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.act_limit = act_limit

        # Initialize counters
        self.global_step = 0
        self.epoch = 0
        self.start_time = None

        # Update hyperparameters to match cleanRL
        self.actor.noise_scale = exploration_noise

    def save(self, filename: str) -> None:
        new_path = os.path.join(save_model_path, filename + ('.pth' if not filename.endswith('.pth') else ''))
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, new_path)

    def receive_trajectory(self, trajectory: RL4SysTrajectory) -> bool:
        if self.start_time is None:
            self.start_time = time.time()

        update = False
        ep_ret, ep_len = 0.0, 0

        for i, r4a in enumerate(trajectory):
            self.global_step += 1
            ep_ret += r4a.rew
            ep_len += 1

            # Store transition in buffer
            self.replay_buffer.store(r4a.obs, r4a.act, r4a.rew, r4a.done)

            # Update handling
            if self.global_step >= self.learning_starts and self.global_step % self.policy_frequency == 0:
                self.train_model()
                update = True

            if r4a.done:
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                ep_ret, ep_len = 0.0, 0

        if update:
            self.epoch += 1
            self.log_epoch()

        return update

    def train_model(self) -> None:
        # Sample from replay buffer
        data, _ = self.replay_buffer.get(self.batch_size)
        obs = data['obs']
        act = data['act']
        rew = data['rew']
        next_obs = data['next_obs']
        done = data['done']

        # Update Q-function (similar to cleanRL)
        with torch.no_grad():
            next_state_actions = self.actor_target(next_obs)
            q_next_target = self.critic_target(next_obs, next_state_actions)
            next_q_value = rew + (1 - done) * self.gamma * q_next_target

        current_q = self.critic(obs, act)
        q_loss = F.mse_loss(current_q, next_q_value)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy (delayed)
        if self.global_step % self.policy_frequency == 0:
            # Actor loss
            actor_loss = -self.critic(obs, self.actor(obs)).mean()
            
            self.pi_optimizer.zero_grad()
            actor_loss.backward()
            self.pi_optimizer.step()

            # Update target networks
            with torch.no_grad():
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # Log losses
            self.logger.store(LossQ=q_loss.item(), LossPi=actor_loss.item())

    def log_epoch(self) -> None:
        elapsed = time.time() - self.start_time if self.start_time else 0
        sps = int(self.global_step / elapsed) if elapsed > 0 else 0

        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('SPS', sps)
        self.logger.dump_tabular()
