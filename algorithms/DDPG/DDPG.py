from _common._algorithms.BaseAlgorithm import AlgorithmAbstract
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import time
import os
from utils.conf_loader import ConfigLoader
from protocol.trajectory import RL4SysTrajectory
from torch.utils.tensorboard import SummaryWriter

from .kernel import DDPGActorCritic
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
        buf_size: int,
        seed: int = hyperparams['seed'],
        gamma: float = hyperparams['gamma'],
        tau: float = hyperparams['tau'],
        learning_rate: float = hyperparams['learning_rate'],
        batch_size: int = hyperparams['batch_size'],
        buffer_size: int = hyperparams['buffer_size'],
        learning_starts: int = hyperparams['learning_starts'],
        policy_frequency: int = hyperparams['policy_frequency'],
        noise_scale: float = hyperparams['noise_scale'],
        train_iters: int = hyperparams['train_iters'],  
    ):
        super().__init__()
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create actor-critic module and its target
        self.ac = DDPGActorCritic(input_size, act_dim, act_limit, noise_scale)
        self.ac_target = DDPGActorCritic(input_size, act_dim, act_limit, noise_scale)
        
        # Copy target parameters
        self.ac_target.load_state_dict(self.ac.state_dict())

        # Set up optimizers
        self.pi_optimizer = Adam(self.ac.actor.parameters(), lr=learning_rate)
        self.q_optimizer = Adam(self.ac.critic.parameters(), lr=learning_rate)

        # Set up replay buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=input_size,
            act_dim=act_dim,
            buf_size=buffer_size,
            gamma=gamma
        )

        # Logging
        log_data_dir = os.path.join(env_dir, './logs/rl4sys-ddpg-info')
        log_data_dir = os.path.join(log_data_dir, f"{int(time.time())}__{seed}")
        os.makedirs(log_data_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_data_dir)


        # Save hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.policy_frequency = policy_frequency
        self.act_limit = act_limit
        self.train_iters = train_iters

        # Initialize counters
        self.global_step = 0
        self.epoch = 0
        self.start_time = None
        self.ep_rewards = 0


    def save(self, filename: str) -> None:
        new_path = os.path.join(save_model_path, filename + ('.pth' if not filename.endswith('.pth') else ''))
        torch.save({
            'actor_critic': self.ac.state_dict(),
        }, new_path)

    def receive_trajectory(self, trajectory: RL4SysTrajectory) -> bool:
        if self.start_time is None:
            self.start_time = time.time()

        update = False
        for i, r4a in enumerate(trajectory):
            self.global_step += 1
            self.ep_rewards += r4a.rew

            # Store transition in buffer
            self.replay_buffer.store(r4a.obs, r4a.act, r4a.rew, r4a.done)

            if r4a.done:
                self.writer.add_scalar("charts/reward", self.ep_rewards, self.global_step)
                self.ep_rewards = 0

            # Update handling
            if self.global_step >= self.learning_starts:
                loss_q, q_vals = self.train_model()
                self.epoch += 1
                update = True

                self.writer.add_scalar("losses/td_loss", loss_q, self.global_step)
                self.writer.add_scalar("losses/q_values", q_vals, self.global_step)
                self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)

            

        return update

    def train_model(self) -> None:
        for _ in range(self.train_iters):
            # Sample from replay buffer
            data, _ = self.replay_buffer.get(self.batch_size)
            obs = data['obs']
            act = data['act']
            rew = data['rew']
            next_obs = data['next_obs']
            done = data['done']

            # Update Q-function (similar to cleanRL)
            with torch.no_grad():
                next_state_actions, _ = self.ac_target.get_action(next_obs)  # Unpack tuple
                next_state_actions = torch.as_tensor(next_state_actions)  # Convert numpy to tensor
                q_next_target = self.ac_target.get_value(next_obs, next_state_actions)
                next_q_value = rew + (1 - done) * self.gamma * q_next_target

            current_q = self.ac.get_value(obs, act)
            q_loss = F.mse_loss(current_q, next_q_value)

            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            # Update policy (delayed)
            if self.global_step % self.policy_frequency == 0:
                # Actor loss
                actions, _ = self.ac.get_action(obs)  # Unpack tuple
                actions = torch.as_tensor(actions)  # Convert numpy to tensor
                actor_loss = -self.ac.get_value(obs, actions).mean()
                
                self.pi_optimizer.zero_grad()
                actor_loss.backward()
                self.pi_optimizer.step()

                # Update target networks
                with torch.no_grad():
                    for param, target_param in zip(self.ac.parameters(), self.ac_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                return q_loss.item(), actor_loss.item()
        
    def log_epoch(self) -> None:
        pass
