#!/usr/bin/env python
"""
Baseline：无 RLlib，手工采样 + 推理 + env.step() 计时
"""
import time, gymnasium as gym, torch, torch.nn as nn
from torch.distributions.categorical import Categorical


class TimedEnv(gym.Env):
    def __init__(self, _): self.env = gym.make("LunarLander-v3")
    def reset(self, **k):   return self.env.reset(**k)
    def step(self, a):
        t0 = time.perf_counter()
        o, r, d, t, info = self.env.step(a)
        info["step_time"] = time.perf_counter() - t0
        return o, r, d, t, info
    @property
    def observation_space(self): return self.env.observation_space
    @property
    def action_space(self):      return self.env.action_space


# ---------- 1. 简单策略网络（与 RLlib 默认 TorchPolicy 配置一致） ----------
class SimplePolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    @torch.no_grad()
    def act(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits).sample().item()

# ---------- 2. Baseline 计时循环 ----------
def run_baseline(num_steps: int = 4000):
    env = TimedEnv(None)
    policy = SimplePolicy(env.observation_space.shape[0],
                          env.action_space.n)

    obs, _ = env.reset(seed=0)
    env_time = infer_time = 0.0
    t0_total = time.perf_counter()

    for _ in range(num_steps):
        obs_t = torch.from_numpy(obs)
        # 2-1 模型推理
        t0 = time.perf_counter()
        action = policy.act(obs_t)
        infer_time += time.perf_counter() - t0

        # 2-2 环境步进
        #t0 = time.perf_counter()
        obs, _, done, trunc, info = env.step(action)
        env_time += info["step_time"]

        if done or trunc:
            obs, _ = env.reset()

    total_time = (time.perf_counter() - t0_total)
    per_step_ms = total_time * 1000 / num_steps
    env_ms  = env_time   * 1000 / num_steps
    infer_ms = infer_time * 1000 / num_steps
    over_ms  = per_step_ms - env_ms - infer_ms

    print({
        "steps/s": round(1000 / per_step_ms, 1),
        "env_ms":  round(env_ms,   3),
        "infer_ms":round(infer_ms, 3),
        "over_ms": round(over_ms,  3),
    })

if __name__ == "__main__":
    for i in range(20):
        run_baseline()
