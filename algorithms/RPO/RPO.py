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
config_loader = ConfigLoader(algorithm='RPO')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path

class RPO(AlgorithmAbstract):
    def __init__(
        self,
        env_dir: str,
        input_size: int,
        act_dim: int,
        seed: int = hyperparams['seed'],
        learning_rate: float = hyperparams['learning_rate'],
        num_steps: int = hyperparams['num_steps'],
        gamma: float = hyperparams['gamma'],
        gae_lambda: float = hyperparams['gae_lambda'],
        clip_coef: float = hyperparams['clip_coef'],
        ent_coef: float = hyperparams['ent_coef'],
        vf_coef: float = hyperparams['vf_coef'],
        max_grad_norm: float = hyperparams['max_grad_norm'],
        rpo_alpha: float = hyperparams['rpo_alpha'],
        update_epochs: int = hyperparams['update_epochs'],
        num_minibatches: int = hyperparams['num_minibatches'],
        anneal_lr: bool = hyperparams['anneal_lr'],
        total_timesteps: int = hyperparams['total_timesteps_anneal_lr'],
        clip_vloss: bool = hyperparams['clip_vloss'],
        target_kl: float = hyperparams['target_kl'],
    ):
        super().__init__()
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create actor-critic
        self.actor = Actor(input_size, act_dim, rpo_alpha)
        self.critic = Critic(input_size)
        
        # Set up optimizer
        self.optimizer = Adam(list(self.actor.parameters()) + list(self.critic.parameters()), 
                            lr=learning_rate)

        # Instant buffer setup (similar to cleanRL)
        self.obs = torch.zeros((num_steps,) + (input_size,))
        self.actions = torch.zeros((num_steps,) + (act_dim,))
        self.logprobs = torch.zeros(num_steps)
        self.rewards = torch.zeros(num_steps)
        self.dones = torch.zeros(num_steps)
        self.values = torch.zeros(num_steps)
        
        # Buffer position tracker
        self.buffer_pos = 0

        # Logging setup
        log_data_dir = os.path.join(env_dir, './logs/rl4sys-rpo-info')
        log_data_dir = os.path.join(log_data_dir, f"{seed}__{int(time.time())}")
        os.makedirs(log_data_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_data_dir)

        # Save hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.batch_size = num_steps
        self.minibatch_size = num_steps // num_minibatches
        self.num_steps = num_steps

        # Initialize tracking variables
        self.global_step = 0
        self.start_time = None
        self.ep_rewards = 0

        # Save additional hyperparameters for LR annealing
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.total_timesteps = total_timesteps
        self.num_updates = total_timesteps // num_steps
        self.update_count = 0

        # Save additional hyperparameters
        self.clip_vloss = clip_vloss
        self.target_kl = target_kl

    def receive_trajectory(self, trajectory: RL4SysTrajectory) -> bool:
        if self.start_time is None:
            self.start_time = time.time()

        update = False

        for i, r4a in enumerate(trajectory):
            self.global_step += 1
            self.ep_rewards += r4a.rew

            # Store in instant buffer
            self.obs[self.buffer_pos] = torch.FloatTensor(r4a.obs)
            self.actions[self.buffer_pos] = torch.FloatTensor(r4a.act)
            self.logprobs[self.buffer_pos] = r4a.data['log_prob']
            self.rewards[self.buffer_pos] = r4a.rew
            self.dones[self.buffer_pos] = r4a.done
            self.values[self.buffer_pos] = r4a.data['v']
            
            self.buffer_pos += 1

            if r4a.done:
                self.writer.add_scalar("charts/reward", self.ep_rewards, self.global_step)
                self.ep_rewards = 0

            # If buffer is full, update policy
            if self.buffer_pos >= self.num_steps:
                self.buffer_pos = 0
                
                # Anneal learning rate if enabled
                if self.anneal_lr:
                    self.update_count += 1
                    frac = 1.0 - (self.update_count - 1.0) / self.num_updates
                    lrnow = frac * self.learning_rate
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lrnow
                    self.writer.add_scalar("charts/learning_rate", lrnow, self.global_step)
                
                pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs, explained_var = self.train_model()
                # Log additional metrics
                self.writer.add_scalar("losses/approx_kl", approx_kl, self.global_step)
                self.writer.add_scalar("losses/old_approx_kl", old_approx_kl, self.global_step)
                self.writer.add_scalar("losses/clipfrac", clipfracs, self.global_step)
                self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
                self.writer.add_scalar("losses/policy_loss", pg_loss, self.global_step)
                self.writer.add_scalar("losses/value_loss", v_loss, self.global_step)
                self.writer.add_scalar("losses/entropy_loss", entropy_loss, self.global_step)
                self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
                update = True

        return update

    def train_model(self) -> None:
        # Calculate advantages
        advantages = torch.zeros_like(self.rewards)
        lastgaelam = 0
        
        with torch.no_grad():
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_value = self.critic(self.obs[-1])
                    nextnonterminal = 1.0 - self.dones[-1]
                else:
                    next_value = self.values[t + 1]
                    nextnonterminal = 1.0 - self.dones[t + 1]
                
                delta = self.rewards[t] + self.gamma * next_value * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            
            returns = advantages + self.values

        # Flatten the batch
        b_obs = self.obs.reshape((-1,) + self.obs.shape[1:])
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.actions.shape[1:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimize policy for K epochs
        clipfracs = []
        for epoch in range(self.update_epochs):
            # Generate random indices
            indices = np.random.permutation(self.num_steps)
            
            # Do minibatch updates
            for start in range(0, self.num_steps, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = indices[start:end]
                
                _, newlogprob, entropy = self.actor.forward(
                    b_obs[mb_inds], 
                    b_actions[mb_inds]
                )
                newvalue = self.critic(b_obs[mb_inds])

                # Policy loss
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                # Calculate approx_kl for early stopping
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                # RPO policy loss
                mb_advantages = b_advantages[mb_inds]
                # Normalize advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if self.clip_vloss:
                    v_loss_unclipped = ((newvalue - b_returns[mb_inds]) ** 2)
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

            # Early stopping based on KL divergence
            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break
        
        
        
        # Calculate explained variance
        y_pred = b_values.detach().numpy()
        y_true = b_returns.detach().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
        
        return (pg_loss.item(), v_loss.item(), entropy_loss.item(), 
                old_approx_kl.item(), approx_kl.item(), np.mean(clipfracs), explained_var)

    def save(self, filename: str) -> None:
        new_path = os.path.join(save_model_path, filename + ('.pth' if not filename.endswith('.pth') else ''))
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, new_path)

    def log_epoch(self, data: dict) -> None:
        pass
