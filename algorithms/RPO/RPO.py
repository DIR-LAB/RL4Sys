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

        # Set up replay buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=input_size,
            act_dim=act_dim,
            size=num_steps,
            gamma=gamma,
            gae_lambda=gae_lambda
        )

        # Logging setup
        log_data_dir = os.path.join(env_dir, './logs/rl4sys-rpo-info')
        log_data_dir = os.path.join(log_data_dir, f"{seed}__{int(time.time())}")
        os.makedirs(log_data_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_data_dir)

        # Save hyperparameters
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.batch_size = num_steps
        self.minibatch_size = num_steps // num_minibatches

        # Initialize tracking variables
        self.global_step = 0
        self.start_time = None

    def receive_trajectory(self, trajectory: RL4SysTrajectory) -> bool:
        if self.start_time is None:
            self.start_time = time.time()

        update = False
        ep_ret, ep_len = 0.0, 0

        for i, r4a in enumerate(trajectory):
            self.global_step += 1
            ep_ret += r4a.rew
            ep_len += 1

            # Get value estimate
            with torch.no_grad():
                value = self.critic(torch.as_tensor(r4a.obs, dtype=torch.float32))

            # Store transition
            self.replay_buffer.store(
                r4a.obs, 
                r4a.act, 
                r4a.rew, 
                value.item(),
                r4a.data['log_prob']
            )

            if r4a.done:
                self.replay_buffer.finish_path()
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                ep_ret, ep_len = 0.0, 0
                update = True

        if update:
            self.train_model()

        return update

    def train_model(self) -> None:
        data = self.replay_buffer.get()
        
        # Train for update_epochs epochs
        for epoch in range(self.update_epochs):
            # Generate random indices
            indices = np.random.permutation(self.batch_size)
            
            # Do minibatch updates
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = indices[start:end]
                
                _, newlogprob, entropy, newvalue = self.actor.forward(
                    data['obs'][mb_inds], 
                    data['act'][mb_inds]
                )
                newvalue = self.critic(data['obs'][mb_inds])

                # Policy loss
                logratio = newlogprob - data['logp'][mb_inds]
                ratio = logratio.exp()

                # RPO policy loss
                advantages = data['adv'][mb_inds]
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue - data['ret'][mb_inds]) ** 2).mean()

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

                # Log
                self.logger.store(
                    LossPi=pg_loss.item(),
                    LossV=v_loss.item(),
                    Entropy=entropy_loss.item()
                )

    def save(self, filename: str) -> None:
        new_path = os.path.join(save_model_path, filename + ('.pth' if not filename.endswith('.pth') else ''))
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, new_path)
