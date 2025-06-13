#!/usr/bin/env python
"""
Distill a LunarLander‑v3 PPO teacher (8‑64‑64‑4 MLP) into a **linear** student network.

Pipeline
========
1.  Train/Load a PPO teacher with Stable‑Baselines3 (architecture fixed to 64‑64).
2.  Roll out the teacher to collect a dataset of (state, teacher_logits).
3.  Train a linear student by minimizing KL(softmax_T(teacher), softmax_T(student)).
4.  Evaluate teacher and student over multiple episodes.

Usage
-----
Install deps first::

    pip install "gymnasium[box2d]" stable-baselines3 torch

Then run::

    python lunarlander_distill.py                # quick test (teacher loads if exists)
    python lunarlander_distill.py --teach_train_steps 500000  # train teacher ~5e5 steps

Tip:  Adjust --teach_train_steps for higher teacher quality; 1e6–2e6 achieves >200 reward.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from torch.distributions import Categorical

# ------------------------------ Teacher wrapper ---------------------------------

class TeacherPolicy(nn.Module):
    """Light wrapper exposing logits from an SB3 PPO model."""

    def __init__(self, sb3_model: PPO):
        super().__init__()
        self.sb3 = sb3_model
        # Freeze to avoid autograd during distillation
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SB3 policy returns latent features; we re‑route to action_net to get raw logits
        latent_pi, _ = self.sb3.policy.mlp_extractor(x)
        logits = self.sb3.policy.action_net(latent_pi)
        return logits

# ------------------------------ Student network ---------------------------------

class StudentLinear(nn.Module):
    def __init__(self, input_dim: int = 8, output_dim: int = 4):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

# ------------------------------ Helpers -----------------------------------------

def collect_dataset(env: gym.Env, teacher: nn.Module, num_steps: int = 50_000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run teacher policy to collect (state, teacher_logits).

    Returns
    -------
    X : Tensor[num_steps, 8]
    Y : Tensor[num_steps, 4]  (teacher logits)
    """
    states, logits = [], []
    obs, _ = env.reset(seed=0)
    for _ in range(num_steps):
        s = torch.tensor(obs, dtype=torch.float32)
        logit = teacher(s.unsqueeze(0)).squeeze(0)
        action = logit.argmax().item()
        obs, _, terminated, truncated, _ = env.step(action)
        states.append(s)
        logits.append(logit)
        if terminated or truncated:
            obs, _ = env.reset()
    return torch.stack(states), torch.stack(logits)


def train_student(X: torch.Tensor, Y: torch.Tensor, epochs: int = 10, batch_size: int = 256, temperature: float = 2.0) -> StudentLinear:
    student = StudentLinear()
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    criterion = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(epochs):
        perm = torch.randperm(X.size(0))
        epoch_loss = 0.0
        for i in range(0, X.size(0), batch_size):
            idx = perm[i : i + batch_size]
            x_b, y_b = X[idx], Y[idx]
            logits_s = student(x_b) / temperature
            logits_t = y_b / temperature
            loss = criterion(
                nn.functional.log_softmax(logits_s, dim=-1),
                nn.functional.softmax(logits_t, dim=-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_b.size(0)
        print(f"[Distill] epoch {epoch+1:02d}  KL={epoch_loss / X.size(0):.4f}")
    return student


def evaluate(env: gym.Env, policy: nn.Module, episodes: int = 20) -> float:
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward, done, truncated = 0.0, False, False
        while not (done or truncated):
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = policy(s).argmax(dim=-1).item()
            obs, r, done, truncated, _ = env.step(action)
            ep_reward += r
        rewards.append(ep_reward)
    return sum(rewards) / len(rewards)

# ------------------------------ Main entry --------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_file", default="ppo_lunarlander")
    parser.add_argument("--teach_train_steps", type=int, default=200_000,
                        help="Timesteps to train PPO teacher if no checkpoint present")
    parser.add_argument("--dataset_steps", type=int, default=50_000,
                        help="Samples to collect for distillation")
    args = parser.parse_args()

    env = gym.make("LunarLander-v3")

    # 1. Load or train teacher ---------------------------------------------------
    ckpt = Path(args.teacher_file + ".zip")
    if ckpt.exists():
        teacher_sb3 = PPO.load(args.teacher_file, env=env)
        print("[Teacher] Loaded checkpoint", ckpt)
    else:
        print("[Teacher] Training new PPO model …")
        teacher_sb3 = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[64, 64]),
            verbose=1,
        )
        teacher_sb3.learn(total_timesteps=args.teach_train_steps)
        teacher_sb3.save(args.teacher_file)
        print("[Teacher] Training done, model saved to", ckpt)
    teacher = TeacherPolicy(teacher_sb3)

    # 2. Collect dataset ---------------------------------------------------------
    print(f"[Data] Collecting {args.dataset_steps} steps …")
    X, Y = collect_dataset(env, teacher, num_steps=args.dataset_steps)

    # 3. Distill -----------------------------------------------------------------
    student = train_student(X, Y)

    # 4. Evaluate ---------------------------------------------------------------
    teacher_score = evaluate(env, teacher)
    student_score = evaluate(env, student)
    print("\n===== Final scores =====")
    print(f"Teacher avg reward : {teacher_score:8.2f}")
    print(f"Student avg reward : {student_score:8.2f}")


if __name__ == "__main__":
    main()
