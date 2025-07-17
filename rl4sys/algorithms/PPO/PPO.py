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
from rl4sys.utils.util import StructuredLogger

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
                 traj_per_epoch: int = 25600,  # full episode (JOB_SEQUENCE_SIZE*100) before update
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
        self.gamma = gamma
        self.lam = lam
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
        self.version = version
        self.seed = seed
        self.type = "onpolicy"
        self.act_dim = act_dim  # Store action dimension for mask creation
        
        # Set device (use GPU if available to match reference implementation)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create normal PPO model
        self._model_train = RLActorCritic(input_size, act_dim, actor_type='kernel').to(self.device) # TODO manually control actor type

        self.models = {}
        # Initialize lock for thread-safe model updates
        self.lock = threading.RLock()
        self.models[self.version] = self._model_train

        # ------------------------------------------------------------------
        # Optimizers (separate policy and value networks as in reference impl.)
        # ------------------------------------------------------------------
        # Policy (actor) optimizer – exclude critic parameters
        self.actor_optimizer = Adam(
            self._model_train.pi.parameters(),
            lr=self._pi_lr,
        )

        # Value (critic) optimizer – operates only on critic network
        self.critic_optimizer = Adam(self._model_train.v.parameters(), lr=self._vf_lr)
        
        # Storage for trajectories
        self.storage_obs = []
        self.storage_act = []
        self.storage_logp = []
        self.storage_rew = []
        self.storage_val = []
        self.storage_done = []
        self.storage_next_obs = []
        self.storage_mask = []  # Add storage for masks
        self.storage_version = [] # Add storage for versions consistency
        
        # Metrics
        self.ep_rewards = 0
        self.train_ep_rewards = 0
        self.start_time = None
        self.traj = 0
        self.epoch = 0
        self.global_step = 0
        
        # Set up loggers
        self.logger = StructuredLogger(f"PPO-{version}", debug=True)
        log_data_dir = os.path.join('./logs/rl4sys-ppo-info', f"{int(time.time())}__{self.seed}")
        os.makedirs(log_data_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_data_dir)
        
        # Set up model save path
        self.save_model_path = os.path.join(log_data_dir, 'models')
        os.makedirs(self.save_model_path, exist_ok=True)
        
        # Log model configuration
        self.logger.info("PPO initialized with normal RLActorCritic", 
                        model_name=self._model_train.get_model_name(),
                        input_size=input_size,
                        act_dim=act_dim)
        

    def save(self, filename: str) -> None:
        """Save the current training model.

        Uses .pth file extension.

        Args:
            filename: name to save file as
        """
        new_path = os.path.join(self.save_model_path, filename +
                                ('.pth' if not filename.__contains__('.pth') else ''))
        torch.save(self._model_train, new_path)

    def receive_trajectory(self, trajectory: RL4SysTrajectory, version: int) -> bool:
        """
        Process a trajectory from the environment.
        
        Args:
            trajectory: Trajectory from environment
            
        Returns:
            bool: True if we just finished an epoch (implies new model)
        """
        # Ignore trajectories generated by older models (stale on-policy data)
        if version != self.version:
            self.logger.debug("Discarding stale trajectory", traj_version=version, current_version=self.version)
            return False

        if self.start_time is None:
            self.start_time = time.time()

        self.storage_version.append(version)
        # Process each step in the trajectory
        for i, r4a in enumerate(trajectory):
            # print(f"r4a: {r4a.obs} {r4a.act} {r4a.rew}")
            #print(r4a.version)
            self.traj += 1
            self.global_step += 1
            self.ep_rewards += r4a.rew
            self.train_ep_rewards += r4a.rew
            obs_value = self._model_train.get_value(torch.as_tensor(r4a.obs, dtype=torch.float32).to(self.device))
            # Store transition
            self.storage_obs.append(np.copy(r4a.obs))
            self.storage_act.append(np.copy(r4a.act))
            self.storage_logp.append(np.copy(r4a.data['logp_a']))
            self.storage_rew.append(np.copy(r4a.rew))
            self.storage_val.append(np.copy(obs_value.detach().cpu().numpy()))
            self.storage_done.append(r4a.done)
            
            # Store mask if available, otherwise use ones
            #print(f"r4a.mask: {r4a.mask}")
            if r4a.mask is not None:
                self.storage_mask.append(np.copy(r4a.mask))
            else:
                # Create a mask of ones with the same shape as action space
                mask = np.ones(self.act_dim)
                self.storage_mask.append(mask)
            
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

            print(f"storage_version: {self.storage_version}")
            pg_loss, v_loss, entropy_loss, approx_kl, clipfracs, explained_var = self.train_model()
            self.epoch += 1
            self.writer.add_scalar("losses/pg_loss", pg_loss, self.global_step)
            self.writer.add_scalar("losses/v_loss", v_loss, self.global_step)
            self.writer.add_scalar("losses/entropy_loss", entropy_loss, self.global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl, self.global_step)
            self.writer.add_scalar("losses/clipfracs", clipfracs, self.global_step)
            self.writer.add_scalar("losses/explained_var", explained_var, self.global_step)
            self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
            self.writer.add_scalar("charts/avg_reward", self.train_ep_rewards/100, self.epoch) # TODO for job-scheduling only
            self.train_ep_rewards = 0
            self.logger.info("PPO training epoch completed", 
                           epoch=self.epoch, pg_loss=pg_loss, v_loss=v_loss, 
                           entropy_loss=entropy_loss, approx_kl=approx_kl, 
                           clipfracs=clipfracs, explained_var=explained_var)
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
        self.storage_mask = []
        self.storage_version = []

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
        masks = np.array(self.storage_mask, dtype=np.float32)
        
        # Convert to tensors and move to device
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        logprobs_tensor = torch.tensor(logprobs, dtype=torch.float32).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(self.device)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        masks_tensor = torch.tensor(masks, dtype=torch.float32).to(self.device)
        
        # Calculate advantages and returns
        with torch.no_grad():
            next_values = self._model_train.get_value(next_obs_tensor)
            advantages = torch.zeros_like(rewards_tensor) 
            lastgaelam = 0
            
            # GAE calculation
            for t in reversed(range(len(rewards))):

                # For the last timestep we bootstrap with ``next_values`` (value of
                # the observation *after* the final stored transition).  For all
                # earlier timesteps we use the value estimate of the next stored
                # state.  This mirrors the CleanRL / reference implementation and
                # avoids the out-of-bounds indexing bug that was introduced when the
                # special-case branch was mistakenly removed.

                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - dones_tensor[t].float()
                    nextvalues = next_values[t]
                else:
                    nextnonterminal = 1.0 - dones_tensor[t].float() # must be [t], [t+1] won't work.
                    #nextvalues = values_tensor[t + 1]
                    nextvalues = next_values[t]
                
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
        b_masks = masks_tensor.reshape(-1, self.act_dim)  # Reshape masks to match action dimension
        
        # Normalize advantages once across the full batch (reference behaviour)
        if self._norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        clipfracs = []

        # ------------------------------------------------------------------
        # 1. Policy (actor) update – multiple gradient steps
        # ------------------------------------------------------------------
        for i in range(self._train_pi_iters):
            # Recompute distribution for the full batch each iteration (no mini-batching)
            pi_dist = self._model_train.pi._distribution(b_obs, mask=b_masks)
            new_logp = pi_dist.log_prob(b_actions)

            logratio = new_logp - b_logprobs
            ratio = torch.exp(logratio)

            # Surrogate losses
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio) * b_advantages
            pg_loss = -torch.mean(torch.min(surr1, surr2))

            # Diagnostics --------------------------------------------------
            approx_kl = torch.mean(b_logprobs - new_logp).item()
            clipfracs.append(((ratio - 1.0).abs() > self._clip_ratio).float().mean().item())

            # Gradient step (actor only)
            self.actor_optimizer.zero_grad()
            pg_loss.backward()
            # Gradient clipping is optional.  To replicate the converging reference
            # implementation we disable it when ``_max_grad_norm`` is set to a value
            # less than or equal to zero.
            if self._max_grad_norm and self._max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self._model_train.pi.parameters(), self._max_grad_norm)
            self.actor_optimizer.step()

            # Early stopping based on KL divergence
            if self._target_kl is not None and approx_kl > 1.5 * self._target_kl:
                break

        # Entropy diagnostics (not used in loss because ent_coef is typically 0)
        entropy_loss = pi_dist.entropy().mean().item()

        # ------------------------------------------------------------------
        # 2. Value function (critic) update
        # ------------------------------------------------------------------
        for _ in range(self._train_v_iters):
            value_pred = self._model_train.get_value(b_obs).view(-1)
            v_loss = torch.mean((b_returns - value_pred) ** 2)

            self.critic_optimizer.zero_grad()
            v_loss.backward()
            if self._max_grad_norm and self._max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self._model_train.v.parameters(), self._max_grad_norm)
            self.critic_optimizer.step()
        
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
            self.traj = 0

        

        # Store metrics for logging
        return pg_loss.item(), v_loss.item(), entropy_loss, approx_kl, float(np.mean(clipfracs)) if clipfracs else 0.0, explained_var
