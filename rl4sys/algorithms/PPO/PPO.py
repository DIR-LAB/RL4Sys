# Standard library imports
import os
import threading
import time
import random
import numpy as np
import copy

# Third-party imports
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# Local imports
from rl4sys.algorithms.PPO.kernel import RLActorCritic
from rl4sys.common.trajectory import RL4SysTrajectory
from rl4sys.utils.logger import EpochLogger, setup_logger_kwargs

class PPO():
    """
    PPO implementation matching the CleanRL version.
    """

    def __init__(self, 
                 version: int,
                 seed: int = 0,
                 input_size: int = 1,
                 act_dim: int = 1,
                 buf_size: int = 1000000,
                 batch_size: int = 64,
                 traj_per_epoch: int = 128,
                 clip_ratio: float = 0.2,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 pi_lr: float = 3e-4,
                 vf_lr: float = 1e-3,
                 train_pi_iters: int = 10,
                 train_v_iters: int = 10,
                 target_kl: float = 0.015,
                 max_grad_norm: float = 0.5,
                 norm_adv: bool = True,
                 clip_vloss: bool = True,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 ):
        
        # Hyperparameters
        self._buf_size = buf_size
        self._batch_size = batch_size
        self._traj_per_epoch = traj_per_epoch
        self._clip_ratio = clip_ratio
        self._gamma = gamma
        self._lam = lam
        self._pi_lr = pi_lr
        self._vf_lr = vf_lr
        self._train_pi_iters = train_pi_iters
        self._train_v_iters = train_v_iters
        self._target_kl = target_kl
        self._max_grad_norm = max_grad_norm
        self._norm_adv = norm_adv
        self._clip_vloss = clip_vloss
        self._ent_coef = ent_coef
        self._vf_coef = vf_coef
        self.gamma = gamma
        self.lam = lam
        self.version = version
        self.seed = seed
        self.type = "onpolicy"

        self.models = {}  # Store model versions for serving
        # Initialize lock for thread-safe model updates
        self.lock = threading.RLock()
        
        # Create actor-critic model for training
        self._model_train = RLActorCritic(input_size, act_dim)
        self.models[self.version] = copy.deepcopy(self._model_train)  # Store initial model version
        
        # Single optimizer for training
        self.optimizer = Adam(self._model_train.parameters(), lr=pi_lr, eps=1e-5)

        # Set up logger
        log_data_dir = os.path.join('./logs/rl4sys-ppo-info', f"{int(time.time())}__{self.seed}")
        os.makedirs(log_data_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_data_dir)

        # Storage buffers
        self.storage_obs = []
        self.storage_act = []
        self.storage_logp = []
        self.storage_rew = []
        self.storage_val = []
        self.storage_done = []
        self.storage_next_obs = []  # Added to match CleanRL's approach

        self.traj = 0
        self.epoch = 0
        self.global_step = 0
        self.start_time = None
        self.ep_rewards = 0


    def save(self, filename: str) -> None:
        """Save the current training model.

        Uses .pth file extension.

        Args:
            filename: name to save file as
        """
        new_path = os.path.join(self.save_model_path, filename +
                                ('.pth' if not filename.__contains__('.pth') else ''))
        torch.save(self._model_train, new_path)

    def receive_trajectory(self, trajectory: RL4SysTrajectory) -> bool:
        """
        Process a trajectory from the environment.
        
        Args:
            trajectory: Trajectory from environment
            
        Returns:
            bool: True if we just finished an epoch (implies new model)
        """
        if self.start_time is None:
            self.start_time = time.time()
            
        # Process each step in the trajectory
        for i, r4a in enumerate(trajectory):
            # print(f"r4a: {r4a.obs} {r4a.act} {r4a.rew}")
            self.traj += 1
            self.global_step += 1
            self.ep_rewards += r4a.rew
            obs_value = self._model_train.get_value(torch.from_numpy(r4a.obs))
            # Store transition
            self.storage_obs.append(np.copy(r4a.obs))
            self.storage_act.append(np.copy(r4a.act))
            self.storage_logp.append(np.copy(r4a.data['logp_a']))
            self.storage_rew.append(np.copy(r4a.rew))
            self.storage_val.append(np.copy(obs_value.detach().numpy()))
            self.storage_done.append(r4a.done)
            
            # Store next observation for bootstrapping
            if i < len(trajectory) - 1:
                self.storage_next_obs.append(trajectory[i+1].obs)
            else:
                # For the last step, use the same observation if not done
                # or zeros if done
                if not r4a.done:
                    self.storage_next_obs.append(r4a.obs)
                else:
                    self.storage_next_obs.append(np.zeros_like(r4a.obs))
            
            self.writer.add_scalar('charts/VVals', obs_value, self.global_step)

            if r4a.done:
                self.writer.add_scalar("charts/reward", self.ep_rewards, self.global_step)
                self.ep_rewards = 0
            
        # Once we have enough trajectories, do an update
        if self.traj > 0 and self.traj > self._traj_per_epoch:
            print(f"\n-----[PPO] Training model for epoch {self.epoch}-----\n")
            pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, explained_var = self.train_model()
            self.epoch += 1
            self.writer.add_scalar("losses/pg_loss", pg_loss, self.global_step)
            self.writer.add_scalar("losses/v_loss", v_loss, self.global_step)
            self.writer.add_scalar("losses/entropy_loss", entropy_loss, self.global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl, self.global_step)
            self.writer.add_scalar("losses/clipfracs", clipfracs, self.global_step)
            self.writer.add_scalar("losses/explained_var", explained_var, self.global_step)
            self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
            print(f"\n-----[PPO] Training model for epoch {self.epoch} completed-----\n")
            return True  # indicates an updated model
        return False

    def get_current_model(self):
        with self.lock:
            return self.models[self.version], self.version

    def _clear_storage(self):
        """Clear all storage arrays after training."""
        self.storage_obs = []
        self.storage_act = []
        self.storage_logp = []
        self.storage_rew = []
        self.storage_val = []
        self.storage_done = []
        self.storage_next_obs = []

    def train_model(self) -> None:
        """
        Train the model using the CleanRL PPO approach.
        """
        # Convert stored transitions to numpy arrays
        obs = np.array(self.storage_obs, dtype=np.float32)
        actions = np.array(self.storage_act, dtype=np.int64)
        logprobs = np.array(self.storage_logp, dtype=np.float32)
        rewards = np.array(self.storage_rew, dtype=np.float32)
        values = np.array(self.storage_val, dtype=np.float32)
        dones = np.array(self.storage_done, dtype=np.bool_)
        next_obs = np.array(self.storage_next_obs, dtype=np.float32)
        
        # Convert to tensors and move to device
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        logprobs_tensor = torch.tensor(logprobs, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        values_tensor = torch.tensor(values, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.bool)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)
        
        # Calculate advantages and returns
        with torch.no_grad():
            next_values = self._model_train.get_value(next_obs_tensor)
            advantages = torch.zeros_like(rewards_tensor) 
            lastgaelam = 0
            
            # GAE calculation
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones_tensor[t].float()
                    nextvalues = next_values[t]
                else:
                    nextnonterminal = 1.0 - dones_tensor[t+1].float()
                    nextvalues = values_tensor[t + 1]
                
                delta = rewards_tensor[t] + self.gamma * nextvalues * nextnonterminal - values_tensor[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            
            returns = advantages + values_tensor
        
        # Flatten the batch
        b_obs = obs_tensor.reshape(-1, obs.shape[-1])
        b_actions = actions_tensor.reshape(-1)
        b_logprobs = logprobs_tensor.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_tensor.reshape(-1)
        
        # Optimize policy and value networks
        batch_size = len(b_obs)
        minibatch_size = max(1, self._batch_size)
        b_inds = np.arange(batch_size)
        clipfracs = []
        
        for epoch in range(self._train_pi_iters):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = self._model_train.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                
                # Policy loss
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # Calculate approx_kl
                    old_approx_kl = (-logratio).mean().item()
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs += [((ratio - 1.0).abs() > self._clip_ratio).float().mean().item()]
                
                # Normalize advantages
                mb_advantages = b_advantages[mb_inds]
                if self._norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # PPO policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if self._clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self._clip_ratio,
                        self._clip_ratio,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - self._ent_coef * entropy_loss + v_loss * self._vf_coef
                
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model_train.parameters(), self._max_grad_norm)
                self.optimizer.step()
            
            # Early stopping based on KL divergence
            if self._target_kl is not None and approx_kl > 1.5 * self._target_kl:
                break
        
        # Calculate explained variance for logging
        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # finish training, update model version
        with self.lock:
            self.version += 1
            self.models[self.version] = copy.deepcopy(self._model_train)  # Store new model version

        # Clear storage arrays after training
        self._clear_storage()

        # Store metrics for logging
        return pg_loss.item(), v_loss.item(), entropy_loss.item(), approx_kl, np.mean(clipfracs), explained_var
