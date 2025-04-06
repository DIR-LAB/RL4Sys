import gymnasium as gym
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import random
import os

from torch.utils.tensorboard import SummaryWriter
import datetime

# ---------------------
# Utilities
# ---------------------

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    if np.isscalar(shape):
        return (length, shape)
    return (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector [x0, x1, x2]
    output:
        [x0 + discount*x1 + discount^2*x2,
         x1 + discount*x2,
         x2]
    """
    return np.array([
        sum(discount**i * x[j] for i, j in enumerate(range(idx, len(x)))) 
        for idx in range(len(x))
    ])

# ---------------------
# Replay Buffer
# ---------------------

class ReplayBuffer:
    """
    A simple FIFO buffer for DQN.
    """
    def __init__(self, obs_dim, mask_dim, buf_size, gamma):
        self.obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros(combined_shape(buf_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(buf_size, dtype=np.int32)
        self.mask_buf = np.zeros(combined_shape(buf_size, mask_dim), dtype=np.float32)
        self.rew_buf = np.zeros(buf_size, dtype=np.float32)
        self.ret_buf = np.zeros(buf_size, dtype=np.float32)

        self.gamma = gamma
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buf_size

    """
    def store(self, obs, act, mask, rew):
        if self.ptr < self.max_size:
            self.obs_buf[self.ptr] = obs
            self.act_buf[self.ptr] = act
            self.mask_buf[self.ptr] = mask
            self.rew_buf[self.ptr] = rew

            # Write the 'next_obs' of the previous step
            if self.ptr > 0:
                self.next_obs_buf[self.ptr - 1] = obs 

            self.ptr += 1
        else:
            # If buffer is full, do a simple shift (FIFO) so we always keep the latest data
            self.obs_buf[:-1] = self.obs_buf[1:]
            self.act_buf[:-1] = self.act_buf[1:]
            self.mask_buf[:-1] = self.mask_buf[1:]
            self.rew_buf[:-1] = self.rew_buf[1:]
            self.next_obs_buf[:-1] = self.next_obs_buf[1:]

            # Insert the new transition at the last position
            last_idx = self.max_size - 1
            self.obs_buf[last_idx] = obs
            self.act_buf[last_idx] = act
            self.mask_buf[last_idx] = mask
            self.rew_buf[last_idx] = rew

            self.next_obs_buf[last_idx - 1] = obs
    """
    def store(self, obs, next_obs, act, mask, rew):
        # Use the same index for both obs and next_obs
        idx = self.ptr % self.max_size  # or whatever indexing logic you like

        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.act_buf[idx] = act
        self.mask_buf[idx] = mask
        self.rew_buf[idx] = rew

        self.ptr += 1


    def finish_path(self, last_val=0):
        """
        At the end of a trajectory, fill in the reward-to-go (ret) for each step in [path_start_idx, ptr).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self, batch_size):
        """
        Sample a random batch of size 'batch_size' from the buffer.
        Returns: dictionary of torch tensors for obs, next_obs, act, rew, ret
        """
        assert self.ptr >= batch_size, "Not enough data in buffer to sample"
        idxs = random.sample(range(len(self.obs_buf)), batch_size)
        data = dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    act=self.act_buf[idxs],
                    mask=self.mask_buf[idxs],
                    rew=self.rew_buf[idxs],
                    ret=self.ret_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

# ---------------------
# DQN Network
# ---------------------

class DeepQNetwork(nn.Module):
    """
    Simple feedforward neural net for approximating the Q-function.
    Epsilon-greedy is handled here in the 'step' method.
    """
    def __init__(self, obs_dim, act_dim, epsilon=1.0, epsilon_min=0.01, epsilon_decay=5e-4):
        super().__init__()

        # For LunarLander-v2, obs_dim=8
        """
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, act_dim)
        )
        """
        

        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def forward(self, obs):
        """
        Forward pass. 
        obs shape: [batch_size, obs_dim]
        returns: Q-values for each action (shape [batch_size, act_dim]).
        """
        return self.q_network(obs)

    def step(self, obs):
        """
        Epsilon-greedy action selection for a single observation (no batch).
        obs shape: [obs_dim]
        returns: (action, debug_info)
        """
        if random.random() < self.epsilon:
            # Random action
            q_vals = self.forward(obs.unsqueeze(0))  # just for logging
            act = random.randrange(q_vals.shape[-1])
        else:
            # Greedy action
            q_vals = self.forward(obs.unsqueeze(0))  # shape [1, act_dim]
            act = q_vals.argmax(dim=1).item()

        # Decay epsilon
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        return act, {'q_val': q_vals.squeeze(0).detach().numpy(), 'epsilon': self.epsilon}

# ---------------------
# DQN Agent
# ---------------------

class DQNAgent:
    def __init__(
        self,
        env_name='LunarLander-v2',
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=5e-4,
        q_lr=1e-3,
        buf_size=10000,
        batch_size=64,
        train_q_iters=1,
        seed=0,
    ):
        """
        Args:
            env_name (str): e.g. 'LunarLander-v2'
            gamma (float): discount factor
            epsilon (float): initial epsilon for epsilon-greedy policy
            epsilon_min (float): min epsilon
            epsilon_decay (float): how much epsilon reduces per step
            q_lr (float): learning rate for Q-network
            buf_size (int): replay buffer size
            batch_size (int): how many experiences to sample per training step
            train_q_iters (int): how many gradient updates per training call
            seed (int): random seed
        """
        # Get path to script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Generate a timestamped subdirectory for logs
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unique_log_dir = os.path.join(script_dir, 'logs', timestamp)
        os.makedirs(unique_log_dir, exist_ok=True)

        # Initialize a TensorBoard writer pointing to our unique logs path
        self.writer = SummaryWriter(log_dir=unique_log_dir)

        # Set random seed
        #seed += 10000 * os.getpid()  # to differentiate seeds if multi-process
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        self.env = gym.make(env_name)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.n

        # Replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim, obs_dim, buf_size, gamma)


        # DQN model
        self.model = DeepQNetwork(obs_dim=obs_dim,
                                  act_dim=act_dim,
                                  epsilon=epsilon,
                                  epsilon_min=epsilon_min,
                                  epsilon_decay=epsilon_decay)
        # Target Q-network (periodically synced to q_main)
        self.q_target = DeepQNetwork(obs_dim=obs_dim,
                                  act_dim=act_dim,
                                  epsilon=epsilon,
                                  epsilon_min=epsilon_min,
                                  epsilon_decay=epsilon_decay)
        self.q_target.load_state_dict(self.model.state_dict())

        self.target_update_freq = 500  # how often to sync weights
        self.total_steps = 0


        # Optimizer
        self.q_optimizer = Adam(self.model.parameters(), lr=q_lr)
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_q_iters = train_q_iters

        self.episode = 0

    def train(self, num_episodes=500):
        """
        Main training loop: run episodes, collect data, update Q-network.
        """
        for ep in range(num_episodes):
            # new gym.reset in Gym >= 0.26 returns (obs, info) 
            # older gym versions just return obs
            reset_out = self.env.reset(seed= self.seed+num_episodes)
            #reset_out = self.env.reset(seed= self.seed+num_episodes)
            if isinstance(reset_out, tuple):
                obs, info = reset_out
            else:
                obs, info = reset_out, {}

            done = False
            ep_ret = 0
            ep_len = 0

            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                act, step_data = self.model.step(obs_tensor)

                """
                step_out = self.env.step(act)
                # For Gym >= 0.26, step_out = (next_obs, rew, done, truncated, info)
                # For older Gym, step_out = (next_obs, rew, done, info)
                if len(step_out) == 5:
                    next_obs, rew, done, truncated, info = step_out
                    done = done or truncated
                else:
                    next_obs, rew, done, info = step_out

                # Store in replay buffer
                self.replay_buffer.store(obs, act, np.zeros_like(obs), rew)
                """
                next_obs, rew, done, truncate, info = self.env.step(act)
                self.replay_buffer.store(obs, next_obs, act, np.zeros_like(obs), rew)

                obs = next_obs
                ep_ret += rew
                ep_len += 1

            # Let replay buffer compute returns for that trajectory
            self.replay_buffer.finish_path()

            # Do a training step (or more)
            self.train_model()

            # Log to TensorBoard
            self.writer.add_scalar('training/episode_return', ep_ret, ep)
            #self.writer.add_scalar('training/episode_length', ep_len, ep) 

            print(f"Episode {ep+1}/{num_episodes}: Return={ep_ret:.2f}, Length={ep_len}")
        
        # Done training
        print("Training complete!")
        # Close the writer
        self.writer.close()

    def train_model(self):
        """
        Sample a batch from replay buffer and do several gradient updates on the Q-network.
        """
        # If we don't have enough data in the buffer yet, skip
        if self.replay_buffer.ptr < self.batch_size:
            return

        for _ in range(self.train_q_iters):
            data = self.replay_buffer.get(self.batch_size)
            loss_q = self.compute_loss_q(data)

            self.q_optimizer.zero_grad()
            loss_q.backward()
            self.q_optimizer.step()

        # ADDED: Periodically update target network
            self.total_steps += 1
            if self.total_steps % self.target_update_freq == 0:
                self.q_target.load_state_dict(self.model.state_dict())

    def compute_loss_q(self, data):
        """
        Compute mean-squared error between current Q-values and target Q-values.
        """
        obs, next_obs = data['obs'], data['next_obs']
        acts, rews = data['act'], data['rew']
        
        # Current Q
        q_values = self.model(obs)  # shape [batch_size, act_dim]
        q_taken = q_values.gather(1, acts.long().unsqueeze(-1)).squeeze(-1)

        # Target Q
        with torch.no_grad():
            #q_next_values = self.model(next_obs)  # [batch_size, act_dim] # single DQN
            q_next_values = self.q_target(next_obs) # ADDED
            q_next_max = q_next_values.max(dim=1)[0]
            q_target = rews + self.gamma * q_next_max

        loss_q = ((q_taken - q_target)**2).mean()
        return loss_q

    def save(self, filename: str = 'dqn_agent.pth'):
        """
        Save the trained model to file.
        """
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename: str = 'dqn_agent.pth'):
        """
        Load model parameters from file.
        """
        self.model.load_state_dict(torch.load(filename))
        print(f"Model loaded from {filename}")

# ---------------------
# Main Script
# ---------------------

if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    print(env.action_space.n)
    agent = DQNAgent(
        env_name='LunarLander-v3',
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=5e-4,
        q_lr=1e-3,
        buf_size=10000,
        batch_size=64,
        train_q_iters=5,
        seed=6
    )
    # Train for 500 episodes
    agent.train(num_episodes=1000)
    agent.save("dqn_lunarlander.pth")
