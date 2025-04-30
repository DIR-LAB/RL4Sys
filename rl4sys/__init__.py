"""
RL4Sys - Reinforcement Learning for Systems

A framework for applying reinforcement learning to system optimization problems,
with a client-server architecture for distributed training and deployment.
"""

from .client import RL4SysAgent
from .server import MyRLServiceServicer

from .proto import (
    GetModelRequest,
    ModelResponse,
    SendTrajectoriesRequest,
    SendTrajectoriesResponse,
    Trajectory,
    Action,
    RLServiceServicer
)
from .algorithms.PPO.PPO import PPO
from .algorithms.DQN.DQN import DQN

from .utils import (
    ConfigLoader,
    serialize_tensor,
    deserialize_tensor,
    serialize_action,
    deserialize_action,
    serialize_model,
    deserialize_model
)

__version__ = '0.1.0'

__all__ = [
    # Client components
    'RL4SysAgent',
    
    # Server components
    'MyRLServiceServicer',
    
    # Algorithms
    'PPO',
    'DQN',
    
    # Protocol messages
    'GetModelRequest',
    'ModelResponse',
    'SendTrajectoriesRequest',
    'SendTrajectoriesResponse',
    'Trajectory',
    'Action',
    'RLServiceServicer',
    # Utils
    'ConfigLoader',
    'serialize_tensor',
    'deserialize_tensor',
    'serialize_action',
    'deserialize_action',
    'serialize_model',
    'deserialize_model'
]
