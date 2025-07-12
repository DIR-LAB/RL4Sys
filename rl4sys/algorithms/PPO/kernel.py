import torch
from torch.optim import Adam
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

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
    def __init__(self, input_size, act_dim, mlp_hidden_sizes=(32, 32, 32, 32, 32), activation=nn.ReLU, actor_type="mlp", job_features=8):
        super().__init__()
        self.input_size = input_size
        self.act_dim = act_dim
        self.job_features = job_features
        self.actor_type = actor_type

        if self.actor_type == "attn":
            self.att_q = nn.Linear(self.job_features, 32)
            self.att_k = nn.Linear(self.job_features, 32)
            self.att_v = nn.Linear(self.job_features, 32)
            self.att_fc16 = nn.Linear(32, 16)
            self.att_fc8 = nn.Linear(16, 8)
            self.att_out = nn.Linear(8, 1)
        elif self.actor_type == "kernel":
            self.pol_fc1 = nn.Linear(self.job_features, 32)
            self.pol_fc2 = nn.Linear(32, 16)
            self.pol_fc3 = nn.Linear(16, 8)
            self.pol_fc_logits = nn.Linear(8, 1)
        else:
            self.pi = mlp([input_size] + list(mlp_hidden_sizes) + [act_dim], activation)

    

    def _distribution(self, obs, mask=None):

        if self.actor_type == "attn":
            batch = obs.shape[0]
            obs_reshaped = obs.view(batch, self.act_dim, self.job_features)
            logits = self._attention_logits(obs_reshaped)
        elif self.actor_type == "kernel":
            batch = obs.shape[0]
            obs_reshaped = obs.view(batch, self.act_dim, self.job_features)
            logits = self._rl_kernel(obs_reshaped)
        else:
            logits = self.pi(obs)

        #print(f"logits: {logits}")
        #print(f"mask: {mask}")
        if mask is not None:
            # Apply mask by setting masked actions to large negative values
            # This ensures they have near-zero probability
            logits = logits + (1 - mask) * -1e8 
        
        result = Categorical(logits=logits)
        #print(f"result: {result}") 
        #exit()
        return result

    def _rl_kernel(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.pol_fc1(obs))
        x = F.relu(self.pol_fc2(x))
        x = F.relu(self.pol_fc3(x))
        logits = self.pol_fc_logits(x).squeeze(-1)  # (batch, act_dim)
        return logits
    
    def _attention_logits(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Scaled dot-product attention per job producing logits (batch, MAX_QUEUE_SIZE)."""
        # obs_seq: (batch, MAX_QUEUE_SIZE, JOB_FEATURES)
        q = F.relu(self.att_q(obs_seq))  # (B, N, 32)
        k = F.relu(self.att_k(obs_seq))  # (B, N, 32)
        v = F.relu(self.att_v(obs_seq))  # (B, N, 32)

        score = torch.matmul(q, k.transpose(-2, -1))  # (B, N, N)
        score = torch.softmax(score, dim=-1)
        attn_out = torch.matmul(score, v)  # (B, N, 32)

        x = F.relu(self.att_fc16(attn_out))  # (B, N, 16)
        x = F.relu(self.att_fc8(x))          # (B, N, 8)
        logits = self.att_out(x).squeeze(-1)  # (B, N)
        return logits

    def forward(self, obs, act=None, mask=None):
        pi = self._distribution(obs, mask)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a


class RLCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(128, 128), activation=nn.ReLU):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)


class RLActorCritic(nn.Module):
    def __init__(self, input_size: int, act_dim: int, actor_type: str = "kernel") -> None:
        """Initialize Actor-Critic network.

        Args:
            input_size (int): Dimension of flattened observation space.
            act_dim (int): Dimension of the discrete action space.
            actor_type (str, optional): Architecture variant for the actor. Defaults to "kernel".
        """

        super().__init__()

        # Build actor network. Pass *actor_type* explicitly as a keyword argument so
        # it is not mistaken for *mlp_hidden_sizes* in the RLActor signature.
        self.pi: RLActor = RLActor(input_size, act_dim, actor_type=actor_type)

        # Build value (critic) network
        self.v: RLCritic = RLCritic(input_size)

    def step(self, obs, mask=None):
        with torch.no_grad():
            pi = self.pi._distribution(obs, mask)
            #v = self.v(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a)
            # Debug logging removed to avoid performance impact

        action_nd = a.detach().cpu().numpy()
        data_dict = {
            #'v': v.numpy(),
            'logp_a': logp_a.detach().cpu().numpy()
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