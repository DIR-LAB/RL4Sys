import torch
import torch.nn as nn
import numpy as np

"""
Network configurations for DQN
"""

class DeepQNetwork(nn.Module):
    """Neural network for DQN.

    Produces Q-values for actions.
    Uses epsilon-greedy strategy for action exploration-exploitation process.

        Args:
            input_size: input observation dimension (flattened)
            act_dim: number of actions (output layer dimensions)
            epsilon: Initial value for epsilon; exploration rate that is decayed over time.
            epsilon_min: Minimum possible value for epsilon
            epsilon_decay: Decay rate for epsilon
    """
    def __init__(self, input_size: int, act_dim: int,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 5e-4,
                 custom_network: nn.Sequential = None):
        super().__init__()
        if custom_network is None:
            self.q_network = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, act_dim)
            )
        else:
            self.q_network = custom_network

        self.input_size = input_size
        self.act_dim = act_dim

        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay

    def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
        """
            Forward pass through Q-network, outputs Q-values for actions.
        Args:
            obs: current observation
            mask: optional mask tensor to mask certain actions
        Returns:
            Q-values for actions
        """
        q_values = self.q_network(obs)
        if mask is not None:
            # Apply mask by setting masked actions to large negative values
            # This ensures they have the lowest Q-values
            q_values = q_values + (mask - 1) * 1e8
        return q_values

    def step(self, obs: torch.Tensor, mask: torch.Tensor = None):
        """
        If you want to rely entirely on DQN.py's linear_schedule, 
        you can pass the updated epsilon each time and
        override self._epsilon. Otherwise, this can remain as is.
        """
        with torch.no_grad():
            q = self.forward(obs, mask)
        # Epsilon-greedy
        if np.random.rand() <= self._epsilon:
            # For random action, respect the mask
            if mask is not None:
                # Get available actions (where mask == 1)
                available_actions = torch.where(mask == 1)[0]
                if len(available_actions) > 0:
                    a = available_actions[np.random.randint(len(available_actions))].item()
                else:
                    # If no actions available, pick any action
                    a = np.random.randint(self.act_dim)
            else:
                a = np.random.randint(self.act_dim)
        else:
            a = q.argmax().item()

        data_dict = {
            'q_val': q.detach().numpy(),
            'epsilon': self._epsilon
        }
        # Decay (you may skip this if now done at the DQN level)
        self._epsilon = max(self._epsilon * self._epsilon_decay, self._epsilon_min)

        return a, data_dict

    def get_model_name(self):
        return "DQN DeepQNetwork"