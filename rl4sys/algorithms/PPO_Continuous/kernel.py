import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions import TransformedDistribution
from torch.distributions.transforms import TanhTransform, AffineTransform


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class RLActorCont(nn.Module):
    def __init__(self, input_size, act_dim, mlp_hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        self.input_size = input_size
        self.act_dim = act_dim

        self.mu_net = mlp([input_size] + list(mlp_hidden_sizes) + [act_dim], activation)
        # Log std as a parameter vector (state-independent diagonal covariance)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.log_std_min = -20.0
        self.log_std_max = 2.0

    def _distribution(self, obs: torch.Tensor) -> Independent:
        mu = self.mu_net(obs)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        base = Independent(Normal(mu, std), 1)
        # Squash to (0.1, 0.9): tanh -> (-1,1) -> (0,1) -> (0.1,0.9)
        transforms = [
            TanhTransform(cache_size=1),
            AffineTransform(loc=torch.tensor(0.5, device=mu.device), scale=torch.tensor(0.5, device=mu.device)),
            AffineTransform(loc=torch.tensor(0.1, device=mu.device), scale=torch.tensor(0.8, device=mu.device)),
        ]
        return TransformedDistribution(base, transforms)

    def forward(self, obs: torch.Tensor, act: torch.Tensor = None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a


class RLCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(128, 128), activation=nn.ReLU):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.v_net(obs), -1)


class RLActorCriticCont(nn.Module):
    def __init__(self, input_size: int, act_dim: int) -> None:
        super().__init__()
        self.pi: RLActorCont = RLActorCont(input_size, act_dim)
        self.v: RLCritic = RLCritic(input_size)

    def step(self, obs: torch.Tensor, mask=None):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            action = pi.sample()
            logp_a = pi.log_prob(action)
        action_nd = action.detach().cpu().numpy()
        data_dict = {
            'logp_a': logp_a.detach().cpu().numpy()
        }
        return action_nd, data_dict

    def act(self, obs: torch.Tensor):
        return self.step(obs)[0]

    def get_model_name(self):
        return "PPO-Cont RLActorCritic"

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None, mask=None):
        pi = self.pi._distribution(obs)
        if action is None:
            action = pi.sample()
        return action, pi.log_prob(action), pi.entropy(), self.v(obs)



