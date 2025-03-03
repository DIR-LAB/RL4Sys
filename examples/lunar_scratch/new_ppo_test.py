import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import time

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer weights using orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOActorCritic(nn.Module):
    """Combined actor-critic network for PPO."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        
        # Actor network - outputs action logits
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01)
        )
        
        # Critic network - outputs state value estimate
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
    
    def get_value(self, obs):
        """Compute value estimate for given observations."""
        return self.critic(obs).squeeze(-1)
    
    def get_action_and_value(self, obs, action=None):
        """Get action distribution, value estimate, and optionally action log prob."""
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.get_value(obs)

class PPO:
    def __init__(
        self,
        env,
        learning_rate=3e-4,
        num_steps=2048,
        num_envs=1,
        num_epochs=10,
        batch_size=64,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gamma=0.99,
        gae_lambda=0.95,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Initialize actor-critic model
        self.agent = PPOActorCritic(
            self.env.observation_space.shape[0],
            self.env.action_space.n
        ).to(self.device)
        self.optimizer = Adam(self.agent.parameters(), lr=learning_rate)

        # Initialize storage
        self.obs = torch.zeros((num_steps, self.env.observation_space.shape[0])).to(self.device)
        self.actions = torch.zeros((num_steps)).to(self.device)
        self.logprobs = torch.zeros((num_steps)).to(self.device)
        self.rewards = torch.zeros((num_steps)).to(self.device)
        self.dones = torch.zeros((num_steps)).to(self.device)
        self.values = torch.zeros((num_steps)).to(self.device)

    def collect_rollouts(self):
        """Collect environment interactions and store in buffers."""
        next_obs, _ = self.env.reset()
        next_done = False
        
        for step in range(self.num_steps):
            # Store current observation
            self.obs[step] = torch.from_numpy(next_obs).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(
                    torch.from_numpy(next_obs).to(self.device).unsqueeze(0)
                )
            
            # Store action info
            self.actions[step] = action
            self.logprobs[step] = logprob
            self.values[step] = value

            # Execute action in environment
            next_obs, reward, done, truncated, _ = self.env.step(action.cpu().numpy())
            done = done or truncated
            
            # Store reward and done flag
            self.rewards[step] = torch.tensor(reward).to(self.device)
            self.dones[step] = torch.tensor(done).to(self.device)

            # Reset environment if episode ended
            if done:
                next_obs, _ = self.env.reset()
                
        return next_obs

    def compute_advantages(self, next_obs):
        """Compute GAE advantages and returns."""
        with torch.no_grad():
            next_value = self.agent.get_value(
                torch.from_numpy(next_obs).to(self.device).unsqueeze(0)
            )
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            
            # Compute GAE advantages
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - float(next_value)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - float(self.dones[t + 1])
                    nextvalues = self.values[t + 1]
                
                delta = (
                    self.rewards[t] 
                    + self.gamma * nextvalues * nextnonterminal 
                    - self.values[t]
                )
                advantages[t] = lastgaelam = (
                    delta 
                    + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + self.values
            
        return advantages, returns

    def update_policy(self, advantages, returns):
        """Update policy using PPO objective."""
        # Flatten the batch
        b_obs = self.obs.reshape((-1,) + (self.env.observation_space.shape[0],))
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimize policy and value network
        clipfracs = []
        for epoch in range(self.num_epochs):
            # Generate random mini-batch indices
            indices = np.random.permutation(self.num_steps)
            
            for start in range(0, self.num_steps, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get new action distributions and values
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[batch_indices],
                    b_actions[batch_indices]
                )
                
                # Calculate policy loss
                logratio = newlogprob - b_logprobs[batch_indices]
                ratio = logratio.exp()
                
                # Advantage normalization
                mb_advantages = b_advantages[batch_indices]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1 - self.clip_coef,
                    1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = 0.5 * ((newvalue - b_returns[batch_indices]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                clipfracs.append(
                    ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                )

    def learn(self, total_timesteps):
        """Main training loop."""
        start_time = time.time()
        num_updates = total_timesteps // self.num_steps
        
        for update in range(num_updates):
            # Collect rollouts
            next_obs = self.collect_rollouts()
            
            # Compute advantages
            advantages, returns = self.compute_advantages(next_obs)
            
            # Update policy
            self.update_policy(advantages, returns)
            
            # Logging
            if update % 10 == 0:
                print(f"Update {update}/{num_updates}, Total Steps: {(update+1)*self.num_steps}")
                print(f"Mean reward: {self.rewards.mean().item():.2f}")
                print(f"Time elapsed: {time.time() - start_time:.2f}s")
                print("----------------------------------------")

def main():
    # Create environment
    env = gym.make("LunarLander-v3")
    
    # Initialize PPO agent
    ppo = PPO(
        env=env,
        learning_rate=3e-4,
        num_steps=2048,
        num_envs=1,
        num_epochs=10,
        batch_size=64,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )
    
    # Train the agent
    ppo.learn(total_timesteps=1000000)
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()
