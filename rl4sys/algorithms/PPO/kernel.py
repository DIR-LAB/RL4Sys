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
    def __init__(self, input_size, act_dim, hidden_sizes=(32, 16, 8), activation=nn.ReLU):
        super().__init__()
        self.pi = mlp([input_size] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs, mask=None):
        logits = self.pi(obs)
        print(f"logits: {logits}")
        print(f"mask: {mask}")
        if mask is not None:
            # Apply mask by setting masked actions to large negative values
            # This ensures they have near-zero probability
            logits = logits + (1 - mask) * -1e8 # TODO check why this is not going to 0
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, act=None, mask=None):
        pi = self._distribution(obs, mask)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class RLCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64, 64), activation=nn.ReLU):
        super().__init__()
        hidden_sizes = (32, 16, 8)
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)


class RLActorCritic(nn.Module):
    def __init__(self, input_size, act_dim):
        super().__init__()
        # build actor function
        self.pi = RLActor(input_size, act_dim)
        # build value function
        self.v = RLCritic(input_size)

    def step(self, obs, mask=None):
        with torch.no_grad():
            pi = self.pi._distribution(obs, mask)
            #v = self.v(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a)
            # Debug logging removed to avoid performance impact

        action_nd = a.detach().numpy()
        data_dict = {
            #'v': v.numpy(),
            'logp_a': logp_a.detach().numpy()
        }
        return action_nd, data_dict

    def act(self, obs, mask=None):
        return self.step(obs, mask)[0]
    
    def get_model_name(self):
        return "PPO RLActorCritic"
    
    def get_value(self, obs):
        return self.v(obs)
    
    def get_action_and_value(self, obs, action, mask=None):
        pi = self.pi._distribution(obs, mask)
        if action is None:
            action = pi.sample()
        return action, pi.log_prob(action), pi.entropy(), self.v(obs)