import torch
from torch.optim import Adam
import torch.nn as nn
from torch.distributions.categorical import Categorical

"""
Network configurations
"""

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class RLActor(nn.Module):
    def __init__(self, kernel_size, kernel_dim):
        super().__init__()
        self.dense1 = nn.Linear(kernel_dim, 32)
        self.dense2 = nn.Linear(32, 16)
        self.dense3 = nn.Linear(16, 8)
        self.dense4 = nn.Linear(8, 1)

        self.kernel_size = kernel_size
        self.kernel_dim = kernel_dim

    def _distribution(self, obs, mask):
        # kernel_size is MAX_QUEUE_SIZE, kernel_dim is JOB_FEATURES
        x = obs.view(-1, self.kernel_size, self.kernel_dim)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        logits = torch.squeeze(self.dense4(x), -1)
        # logits = self.logits_net(obs)
        logits = logits + (mask-1)*1000000
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, mask, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        x = obs.view(-1, self.kernel_size, self.kernel_dim)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        logits = torch.squeeze(self.dense4(x), -1)
        logits = logits + (mask-1)*1000000
        pi = Categorical(logits=logits)
        # pi = self._distribution(obs, mask)

        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class RLCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64, 64), activation=nn.ReLU):
        super().__init__()
        hidden_sizes = (32, 16, 8)
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, mask):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)


class RLActorCritic(nn.Module):
    def __init__(self, flatten_obs_dim, kernel_size, kernel_dim):
        super().__init__()
        # build actor function
        self.pi = RLActor(flatten_obs_dim, kernel_size, kernel_dim)
        # build value function
        self.v = RLCritic(flatten_obs_dim)

    def step(self, obs, mask):
        with torch.no_grad():
            pi = self.pi._distribution(obs, mask)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs, mask)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
