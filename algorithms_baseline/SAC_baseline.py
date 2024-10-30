# algorithms/sac_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        action_probs = nn.functional.softmax(self.fc3(x), dim=-1)
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action_onehot):
        x = torch.cat([state, action_onehot], dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        q_value = self.fc3(x)
        return q_value

class SACAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.005  # For soft updates
        self.alpha = 0.2  # Entropy coefficient
        self.learning_rate = 1e-3
        self.batch_size = 64

        # Networks
        self.actor = Actor(state_size, action_size)
        self.critic_1 = Critic(state_size, action_size)
        self.critic_2 = Critic(state_size, action_size)
        self.target_critic_1 = Critic(state_size, action_size)
        self.target_critic_2 = Critic(state_size, action_size)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.learning_rate)

        # Replay Buffer
        self.memory = deque(maxlen=100000)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic_1.to(self.device)
        self.critic_2.to(self.device)
        self.target_critic_1.to(self.device)
        self.target_critic_2.to(self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Convert actions to one-hot encoding
        action_onehot = torch.zeros(self.batch_size, self.action_size).to(self.device)
        action_onehot.scatter_(1, actions, 1)

        # Critic loss
        with torch.no_grad():
            next_action_probs = self.actor(next_states)
            next_dist = torch.distributions.Categorical(next_action_probs)
            next_actions = next_dist.sample()
            next_action_onehot = torch.zeros(self.batch_size, self.action_size).to(self.device)
            next_action_onehot.scatter_(1, next_actions.unsqueeze(1), 1)

            target_q1 = self.target_critic_1(next_states, next_action_onehot)
            target_q2 = self.target_critic_2(next_states, next_action_onehot)
            target_q = torch.min(target_q1, target_q2)
            target_value = rewards + (1 - dones) * self.gamma * (target_q - self.alpha * next_dist.log_prob(next_actions).unsqueeze(1))

        current_q1 = self.critic_1(states, action_onehot)
        current_q2 = self.critic_2(states, action_onehot)

        critic_1_loss = nn.functional.mse_loss(current_q1, target_value)
        critic_2_loss = nn.functional.mse_loss(current_q2, target_value)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Actor loss
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        actions = dist.sample()
        action_onehot = torch.zeros(self.batch_size, self.action_size).to(self.device)
        action_onehot.scatter_(1, actions.unsqueeze(1), 1)

        q1 = self.critic_1(states, action_onehot)
        q2 = self.critic_2(states, action_onehot)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * dist.log_prob(actions).unsqueeze(1) - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update(self):
        pass  # No periodic update needed for SAC
