import os
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import random
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = "ppo_baseline3"
    seed: int = 1
    cuda: bool = True
    env_id: str = "LunarLander-v3"
    total_timesteps: int = 500000
    learning_rate: float = 3e-4
    num_envs: int = 4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None


class RewardLoggerCallback(BaseCallback):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer
        self.rewards = [[] for _ in range(4)]  # For 4 environments

    def _on_step(self):
        # Log rewards for each environment
        for i, info in enumerate(self.locals["infos"]):
            if "episode" in info:
                self.writer.add_scalar(f"reward/rewards{i}", info["episode"]["r"], self.num_timesteps)
        return True


def make_env(env_id, idx, capture_video=False, run_name=None):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def main():
    args = Args()
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create vectorized environment
    envs = DummyVecEnv([make_env(args.env_id, i) for i in range(args.num_envs)])
    envs = VecMonitor(envs)

    # Initialize PPO with matching hyperparameters
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=args.learning_rate,
        n_steps=args.num_steps,
        batch_size=int(args.num_envs * args.num_steps),
        n_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_coef,
        clip_range_vf=args.clip_coef if args.clip_vloss else None,
        normalize_advantage=args.norm_adv,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        tensorboard_log=f"runs/{run_name}",
        verbose=1,
        seed=args.seed,
        device="cuda" if args.cuda and torch.cuda.is_available() else "cpu",
    )

    # Custom callback for logging rewards
    reward_callback = RewardLoggerCallback(writer)

    # Train the agent
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=reward_callback,
        progress_bar=True,
    )

    # Close environments and writer
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
