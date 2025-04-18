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
config_loader = ConfigLoader(algorithm='TD3')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path

class TD3(AlgorithmAbstract):
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
        policy_noise: float = hyperparams['policy_noise'],
        noise_clip: float = hyperparams['noise_clip'],
        learning_starts: int = hyperparams['learning_starts'],
        policy_frequency: int = hyperparams['policy_frequency'],
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
        self.actor_optimizer = Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=learning_rate)

        # Set up replay buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=input_size,
            act_dim=act_dim,
            buf_size=buffer_size,
            gamma=gamma
        )

        # Logging setup
        log_data_dir = os.path.join(env_dir, './logs/rl4sys-td3-info')
        log_data_dir = os.path.join(log_data_dir, f"{seed}__{int(time.time())}")
        os.makedirs(log_data_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_data_dir)

        # Save hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.act_limit = act_limit

        # Initialize tracking variables
        self.global_step = 0
        self.start_time = None

        # Set exploration noise
        self.actor.noise_scale = exploration_noise

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
            if self.global_step >= self.learning_starts:
                self.train_model()
                update = True

            if r4a.done:
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                ep_ret, ep_len = 0.0, 0

        if update:
            self.log_epoch()

        return update

    def train_model(self) -> None:
        data, _ = self.replay_buffer.get(self.batch_size)
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(data['act']) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(data['next_obs']) + noise
            ).clamp(-self.act_limit, self.act_limit)
            
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(data['next_obs'], next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = data['rew'] + (1 - data['done']) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(data['obs'], data['act'])

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.global_step % self.policy_frequency == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(data['obs'], self.actor(data['obs'])).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # Log losses
            self.logger.store(
                LossQ=critic_loss.item(),
                LossPi=actor_loss.item()
            )

    def save(self, filename: str) -> None:
        new_path = os.path.join(save_model_path, filename + ('.pth' if not filename.endswith('.pth') else ''))
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, new_path)

    def log_epoch(self) -> None:
        elapsed = time.time() - self.start_time if self.start_time else 0
        sps = int(self.global_step / elapsed) if elapsed > 0 else 0

        self.logger.log_tabular('Epoch', self.global_step)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('SPS', sps)
        self.logger.dump_tabular()
