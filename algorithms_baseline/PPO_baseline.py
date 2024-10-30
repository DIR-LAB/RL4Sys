# algorithms/ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ActorCritic, self).__init__()
        # Common network
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()

        # Actor head
        self.actor_fc = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

        # Critic head
        self.critic_fc = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        action_probs = self.softmax(self.actor_fc(x))
        state_value = self.critic_fc(x)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.learning_rate = 1e-3
        self.K_epochs = 4  # Update policy for K epochs

        # Policy network
        self.policy = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Old policy network
        self.policy_old = ActorCritic(state_size, action_size)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Memory
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.state_values = []

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.policy_old.to(self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_probs, state_value = self.policy_old(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        self.states.append(state.cpu())
        self.actions.append(action.cpu())
        self.logprobs.append(dist.log_prob(action).cpu())
        self.state_values.append(state_value.cpu())

        return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def optimize_model(self):
        pass  # Optimization happens in the update method

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert lists to tensors
        old_states = torch.stack(self.states).to(self.device).detach()
        old_actions = torch.stack(self.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.logprobs).to(self.device).detach()
        old_state_values = torch.stack(self.state_values).squeeze().to(self.device).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate action probabilities and state values
            action_probs, state_values = self.policy(old_states)
            dist = torch.distributions.Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            # Compute advantages
            advantages = rewards - state_values.squeeze().detach()

            # Compute ratio
            ratios = torch.exp(logprobs - old_logprobs)

            # Compute surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Total loss
            loss = -torch.min(surr1, surr2) + 0.5 * nn.functional.mse_loss(state_values.squeeze(), rewards) - 0.01 * dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear memory
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.state_values = []
