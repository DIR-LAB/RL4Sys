from _common._algorithms.BaseAlgorithm import AlgorithmAbstract

import numpy as np
import torch
from torch.optim import Adam

from _common._rl4sys.BaseTrajectory import RL4SysTrajectoryAbstract
from .replay_buffer import ReplayBuffer

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.logger import EpochLogger, setup_logger_kwargs
from trajectory import RL4SysTrajectory

from conf_loader import ConfigLoader
"""Import and load RL4Sys/config.json REINFORCE algorithm configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
config_loader = ConfigLoader(algorithm='REINFORCE')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path


class REINFORCE(AlgorithmAbstract):
    def __init__(self, env_dir: str, kernel_size: int, kernel_dim: int, buf_size: int, seed: int, traj_per_epoch: int, gamma: float, polyak: float, pi_lr: float, q_lr: float, batch_size: i):
        pass

    def save(self, filename) -> None:
        pass

    def receive_trajectory(self, trajectory: RL4SysTrajectoryAbstract) -> bool:
        pass

    def train_model(self) -> None:
        pass

    def log_epoch(self) -> None:
        pass

    def compute_loss(self, data):
        pass
