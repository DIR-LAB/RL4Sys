import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import os
import time
from typing import Tuple, List, Optional, Dict, Any
import scipy.signal


def combined_shape(length: int, shape: Optional[Tuple] = None) -> Tuple:
    """
    Create a combined shape tuple for buffer initialization.
    
    Args:
        length: The length dimension
        shape: Optional additional shape dimensions
        
    Returns:
        Combined shape tuple
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """
    Compute discounted cumulative sum using scipy signal processing.
    
    Args:
        x: Input array
        discount: Discount factor
        
    Returns:
        Discounted cumulative sum
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def mlp(sizes: List[int], activation: nn.Module, output_activation: nn.Module = nn.Identity) -> nn.Sequential:
    """
    Build a multi-layer perceptron.
    
    Args:
        sizes: List of layer sizes
        activation: Activation function
        output_activation: Output activation function
        
    Returns:
        Sequential neural network
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class JobRLActor(nn.Module):
    """
    Job-specific actor network for policy approximation.
    """
    
    def __init__(self, input_size: int, act_dim: int, max_queue_size: int = 256, 
                 job_features: int = 8, use_attention: bool = False, 
                 network_type: str = 'rl_kernel', hidden_sizes: Tuple[int, ...] = (32, 32, 32, 32, 32)):
        """
        Initialize the job-specific actor network.
        
        Args:
            input_size: Input size (max_queue_size * job_features)
            act_dim: Action dimension (will be overridden by actual queue size)
            max_queue_size: Maximum number of jobs in queue
            job_features: Number of features per job
            use_attention: Whether to use attention mechanism
            network_type: Type of network architecture
            hidden_sizes: Hidden layer sizes for MLP variants
        """
        super().__init__()
        self.input_size = input_size
        self.act_dim = act_dim  # This will be overridden by actual queue size
        self.max_queue_size = max_queue_size
        self.job_features = job_features
        self.use_attention = use_attention
        self.network_type = network_type
        
        # Calculate actual queue size from input size
        self.actual_queue_size = input_size // job_features
        
        if use_attention or network_type == 'attention':
            self.policy_net = JobAttentionNetwork(max_queue_size, job_features)
        elif network_type == 'mlp_v1':
            self.policy_net = JobMLPV1(max_queue_size, job_features, self.actual_queue_size)
        elif network_type == 'mlp_v2':
            self.policy_net = JobMLPV2(max_queue_size, job_features, self.actual_queue_size)
        elif network_type == 'mlp_v3':
            self.policy_net = JobMLPV3(max_queue_size, job_features, self.actual_queue_size)
        elif network_type == 'lenet':
            self.policy_net = JobLeNet(max_queue_size, job_features, self.actual_queue_size)
        else:  # default to rl_kernel
            self.policy_net = JobRLKernel(max_queue_size, job_features)
    
    def _distribution(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Categorical:
        """
        Create action distribution from observations.
        
        Args:
            obs: Observation tensor
            mask: Optional action mask
            
        Returns:
            Categorical distribution
        """
        logits = self.policy_net(obs)
        
        # Apply mask if provided (only for rl_kernel and attention networks)
        if mask is not None and self.network_type in ['rl_kernel', 'attention']:
            # Ensure mask has the correct shape
            if mask.shape[-1] != logits.shape[-1]:
                # Resize mask to match actual queue size
                actual_queue_size = logits.shape[-1]
                if mask.shape[-1] > actual_queue_size:
                    mask = mask[..., :actual_queue_size]
                else:
                    # Pad mask with zeros if needed
                    padding_size = actual_queue_size - mask.shape[-1]
                    mask = torch.cat([mask, torch.zeros(mask.shape[:-1] + (padding_size,), device=mask.device)], dim=-1)
            logits = logits + (1 - mask) * -1e8
        
        return Categorical(logits=logits)
    
    def _log_prob_from_distribution(self, pi: Categorical, act: torch.Tensor) -> torch.Tensor:
        """
        Get log probability from distribution.
        
        Args:
            pi: Categorical distribution
            act: Actions
            
        Returns:
            Log probabilities
        """
        return pi.log_prob(act)
    
    def forward(self, obs: torch.Tensor, act: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> Tuple[Categorical, Optional[torch.Tensor]]:
        """
        Forward pass through the actor network.
        
        Args:
            obs: Observation tensor
            act: Optional action tensor
            mask: Optional mask tensor
            
        Returns:
            Tuple of (distribution, log_prob)
        """
        pi = self._distribution(obs, mask)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class JobRLCritic(nn.Module):
    """
    Job-specific critic network for value function approximation.
    """
    
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, ...] = (32, 16, 8)):
        """
        Initialize the job-specific critic network.
        
        Args:
            input_size: Input size
            hidden_sizes: Hidden layer sizes
        """
        super().__init__()
        self.v_net = mlp([input_size] + list(hidden_sizes) + [1], nn.ReLU)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Value estimates
        """
        return torch.squeeze(self.v_net(obs), -1)


class JobRLKernel(nn.Module):
    """
    RL kernel network for policy approximation.
    """
    
    def __init__(self, max_queue_size: int, job_features: int):
        """
        Initialize the RL kernel network.
        
        Args:
            max_queue_size: Maximum number of jobs in queue
            job_features: Number of features per job
        """
        super().__init__()
        self.max_queue_size = max_queue_size
        self.job_features = job_features
        
        self.network = nn.Sequential(
            nn.Linear(job_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RL kernel network.
        
        Args:
            x: Input tensor of shape (batch_size, max_queue_size * job_features)
            
        Returns:
            Policy logits
        """
        batch_size = x.shape[0]
        # Calculate actual queue size from input dimensions
        actual_queue_size = x.shape[1] // self.job_features
        
        # Reshape to (batch_size, actual_queue_size, job_features)
        x = x.view(batch_size, actual_queue_size, self.job_features)
        x = self.network(x)  # (batch_size, actual_queue_size, 1)
        return x.squeeze(-1)  # (batch_size, actual_queue_size)


class JobAttentionNetwork(nn.Module):
    """
    Attention-based network for policy approximation.
    """
    
    def __init__(self, max_queue_size: int, job_features: int):
        """
        Initialize the attention network.
        
        Args:
            max_queue_size: Maximum number of jobs in queue
            job_features: Number of features per job
        """
        super().__init__()
        self.max_queue_size = max_queue_size
        self.job_features = job_features
        
        # Query, Key, Value projections
        self.query = nn.Linear(job_features, 32)
        self.key = nn.Linear(job_features, 32)
        self.value = nn.Linear(job_features, 32)
        
        # Output processing
        self.output_net = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention network.
        
        Args:
            x: Input tensor of shape (batch_size, max_queue_size * job_features)
            
        Returns:
            Policy logits
        """
        batch_size = x.shape[0]
        # Calculate actual queue size from input dimensions
        actual_queue_size = x.shape[1] // self.job_features
        
        # Reshape to (batch_size, actual_queue_size, job_features)
        x = x.view(batch_size, actual_queue_size, self.job_features)
        
        # Compute Q, K, V
        q = self.query(x)  # (batch_size, actual_queue_size, 32)
        k = self.key(x)    # (batch_size, actual_queue_size, 32)
        v = self.value(x)  # (batch_size, actual_queue_size, 32)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # (batch_size, actual_queue_size, actual_queue_size)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, v)  # (batch_size, actual_queue_size, 32)
        
        # Process output
        x = self.output_net(attended)  # (batch_size, actual_queue_size, 1)
        return x.squeeze(-1)  # (batch_size, actual_queue_size)


class JobMLPV1(nn.Module):
    """
    MLP version 1 network architecture.
    """
    
    def __init__(self, max_queue_size: int, job_features: int, act_dim: int):
        """
        Initialize MLP v1 network.
        
        Args:
            max_queue_size: Maximum number of jobs in queue
            job_features: Number of features per job
            act_dim: Action dimension
        """
        super().__init__()
        input_size = max_queue_size * job_features
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP v1.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        return self.network(x)


class JobMLPV2(nn.Module):
    """
    MLP version 2 network architecture.
    """
    
    def __init__(self, max_queue_size: int, job_features: int, act_dim: int):
        """
        Initialize MLP v2 network.
        
        Args:
            max_queue_size: Maximum number of jobs in queue
            job_features: Number of features per job
            act_dim: Action dimension
        """
        super().__init__()
        input_size = max_queue_size * job_features
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, act_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP v2.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        return self.network(x)


class JobMLPV3(nn.Module):
    """
    MLP version 3 network architecture.
    """
    
    def __init__(self, max_queue_size: int, job_features: int, act_dim: int):
        """
        Initialize MLP v3 network.
        
        Args:
            max_queue_size: Maximum number of jobs in queue
            job_features: Number of features per job
            act_dim: Action dimension
        """
        super().__init__()
        input_size = max_queue_size * job_features
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, act_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP v3.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        return self.network(x)


class JobLeNet(nn.Module):
    """
    LeNet-style convolutional network architecture.
    """
    
    def __init__(self, max_queue_size: int, job_features: int, act_dim: int):
        """
        Initialize LeNet network.
        
        Args:
            max_queue_size: Maximum number of jobs in queue
            job_features: Number of features per job
            act_dim: Action dimension
        """
        super().__init__()
        self.max_queue_size = max_queue_size
        self.job_features = job_features
        m = int(np.sqrt(max_queue_size))
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(job_features, 32, kernel_size=1, stride=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate the size after convolutions and pooling
        conv_output_size = 64 * (m // 4) * (m // 4)
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 64),
            nn.Linear(64, act_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LeNet.
        
        Args:
            x: Input tensor of shape (batch_size, max_queue_size * job_features)
            
        Returns:
            Output logits
        """
        batch_size = x.shape[0]
        # Calculate actual queue size from input dimensions
        actual_queue_size = x.shape[1] // self.job_features
        
        # Calculate the square root for reshaping to 2D
        m = int(np.sqrt(actual_queue_size))
        
        # Ensure we have a valid square shape, pad if necessary
        if m * m != actual_queue_size:
            # Pad to the next perfect square
            target_size = (m + 1) * (m + 1)
            padding_size = target_size - actual_queue_size
            x = torch.cat([x, torch.zeros(batch_size, padding_size * self.job_features, device=x.device)], dim=1)
            m = m + 1
        
        # Reshape to (batch_size, channels, height, width)
        x = x.view(batch_size, self.job_features, m, m)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class JobRLActorCritic(nn.Module):
    """
    Job-specific Actor-Critic network combining policy and value function.
    Matches the interface of RLActorCritic from kernel.py.
    """
    
    def __init__(self, input_size: int, act_dim: int, max_queue_size: int = 256, 
                 job_features: int = 8, use_attention: bool = False, network_type: str = 'rl_kernel'):
        """
        Initialize the job-specific actor-critic network.
        
        Args:
            input_size: Input size (max_queue_size * job_features)
            act_dim: Action dimension (will be overridden by actual queue size)
            max_queue_size: Maximum number of jobs in queue
            job_features: Number of features per job
            use_attention: Whether to use attention mechanism
            network_type: Type of network architecture
        """
        super().__init__()
        
        # Build actor function
        self.pi = JobRLActor(input_size, act_dim, max_queue_size, job_features, use_attention, network_type)
        
        # Build value function
        self.v = JobRLCritic(input_size)
        
        # Store the actual action dimension
        self.actual_act_dim = self.pi.actual_queue_size
    
    def step(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Take a step using the current policy.
        
        Args:
            obs: Observation tensor
            mask: Optional action mask
            
        Returns:
            Tuple of (action, data_dict)
        """
        with torch.no_grad():
            pi = self.pi._distribution(obs, mask)
            a = pi.sample()
            logp_a = pi.log_prob(a)
        
        action_nd = a.detach().cpu().numpy()
        data_dict = {
            'logp_a': logp_a.detach().cpu().numpy()
        }
        return action_nd, data_dict
    
    def act(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Get action from policy.
        
        Args:
            obs: Observation tensor
            mask: Optional action mask
            
        Returns:
            Action array
        """
        return self.step(obs, mask)[0]
    
    def get_model_name(self) -> str:
        """
        Get model name.
        
        Returns:
            Model name string
        """
        return "PPO JobRLActorCritic"
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Value estimates
        """
        return self.v(obs)
    
    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None, 
                           mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        
        Args:
            obs: Observation tensor
            action: Optional action tensor
            mask: Optional mask tensor
            
        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        pi = self.pi._distribution(obs, mask)
        if action is None:
            action = pi.sample()
        return action, pi.log_prob(action), pi.entropy(), self.v(obs)


# Keep the PPOBuffer and utility functions for backward compatibility
class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    
    def __init__(self, obs_dim: Tuple, act_dim: Tuple, size: int, gamma: float = 0.99, lam: float = 0.95):
        """
        Initialize the PPO buffer.
        
        Args:
            obs_dim: Observation dimension
            act_dim: Action dimension
            size: Buffer size
            gamma: Discount factor
            lam: GAE lambda parameter
        """
        size = size * 100  # assume the traj can be really long
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.cobs_buf = None
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.mask_buf = np.zeros(combined_shape(size, 256), dtype=np.float32)  # MAX_QUEUE_SIZE = 256
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
    
    def store(self, obs: np.ndarray, cobs: Optional[np.ndarray], act: np.ndarray, 
              mask: np.ndarray, rew: float, val: float, logp: float) -> None:
        """
        Append one timestep of agent-environment interaction to the buffer.
        
        Args:
            obs: Observation
            cobs: Context observation (unused)
            act: Action
            mask: Action mask
            rew: Reward
            val: Value estimate
            logp: Log probability
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.mask_buf[self.ptr] = mask
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
    
    def finish_path(self, last_val: float = 0) -> None:
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        
        Args:
            last_val: Value estimate for the last state
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
    
    def get(self) -> List[np.ndarray]:
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        
        Returns:
            List of buffer data: [obs, act, mask, adv, ret, logp]
        """
        assert self.ptr < self.max_size
        actual_size = self.ptr
        self.ptr, self.path_start_idx = 0, 0
        
        actual_adv_buf = np.array(self.adv_buf, dtype=np.float32)
        actual_adv_buf = actual_adv_buf[:actual_size]
        
        # Normalize advantages
        adv_sum = np.sum(actual_adv_buf)
        adv_n = len(actual_adv_buf)
        adv_mean = adv_sum / adv_n
        adv_sum_sq = np.sum((actual_adv_buf - adv_mean) ** 2)
        adv_std = np.sqrt(adv_sum_sq / adv_n)
        actual_adv_buf = (actual_adv_buf - adv_mean) / adv_std
        
        return [self.obs_buf[:actual_size], self.act_buf[:actual_size], 
                self.mask_buf[:actual_size], actual_adv_buf,
                self.ret_buf[:actual_size], self.logp_buf[:actual_size]]


def count_vars(module: nn.Module) -> int:
    """
    Count the number of parameters in a module.
    
    Args:
        module: PyTorch module
        
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def create_network_architecture(network_type: str, max_queue_size: int, job_features: int, act_dim: int) -> nn.Module:
    """
    Create a network architecture based on the specified type.
    
    Args:
        network_type: Type of network architecture
        max_queue_size: Maximum number of jobs in queue
        job_features: Number of features per job
        act_dim: Action dimension
        
    Returns:
        PyTorch module for the specified architecture
    """
    if network_type == 'rl_kernel':
        return JobRLKernel(max_queue_size, job_features)
    elif network_type == 'attention':
        return JobAttentionNetwork(max_queue_size, job_features)
    elif network_type == 'mlp_v1':
        return JobMLPV1(max_queue_size, job_features, act_dim)
    elif network_type == 'mlp_v2':
        return JobMLPV2(max_queue_size, job_features, act_dim)
    elif network_type == 'mlp_v3':
        return JobMLPV3(max_queue_size, job_features, act_dim)
    elif network_type == 'lenet':
        return JobLeNet(max_queue_size, job_features, act_dim)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
