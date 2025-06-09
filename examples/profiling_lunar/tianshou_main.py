#!/usr/bin/env python
"""Tianshou PPO profiling – LunarLander-v3 (per-step metrics, ts==0.5.1)."""

import time, gymnasium as gym, numpy as np, torch, torch.nn as nn
from collections import defaultdict
from typing import Dict, Any

from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.data import VectorReplayBuffer, Collector
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic


# ------------------------------------------------------------------ #
# 1. 计时环境                                                         #
# ------------------------------------------------------------------ #
class StepTimer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env_time_ns = 0
        self.env_steps   = 0

    def step(self, action):
        t0 = time.perf_counter_ns()
        obs, rew, term, trunc, info = self.env.step(action)
        self.env_time_ns += time.perf_counter_ns() - t0
        self.env_steps   += 1
        return obs, rew, term, trunc, info

    def reset(self, **kw):
        #t0 = time.perf_counter_ns()
        obs, info = self.env.reset(**kw)
        #self.env_time_ns += time.perf_counter_ns() - t0
        return obs, info


# ------------------------------------------------------------------ #
# 2. 组装 Timed PPOPolicy                                            #
# ------------------------------------------------------------------ #
def make_policy(obs_shape: int, n_act: int, timer) -> PPOPolicy:
    backbone = Net(state_shape=obs_shape, hidden_sizes=[64, 64],
                   activation=nn.ReLU, device="cpu")

    actor  = Actor(backbone, action_shape=n_act, device="cpu").to("cpu")
    critic = Critic(Net(state_shape=obs_shape, hidden_sizes=[64, 64],
                        activation=nn.ReLU, device="cpu"),
                    device="cpu").to("cpu")

    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

    policy = PPOPolicy(
        actor, critic, optim,
        action_space=gym.spaces.Discrete(n_act),
        dist_fn=torch.distributions.Categorical,
        vf_coef=0.5, ent_coef=0.0, gae_lambda=0.95,
        max_grad_norm=0.5, eps_clip=0.2, value_clip=True,
        reward_normalization=False,
    )

    # ➊——补丁放回 forward，仅统计 mode='compute_action' ----------
    orig_fwd = policy.forward

    def timed_forward(self_pol, batch, state=None, **kw):
        #if kw.get("mode") != "compute_action":      # 采样阶段才计时
        #    return orig_fwd(batch, state=state, **kw)
        t0 = time.perf_counter_ns()
        out = orig_fwd(batch, state=state, **kw)
        timer["infer_ns"]   += time.perf_counter_ns() - t0
        timer["infer_steps"] += batch.obs.shape[0]
        return out

    policy.forward = timed_forward.__get__(policy, PPOPolicy)   # ➋——重新挂载
    return policy


# ------------------------------------------------------------------ #
# 3. Profiling runner                                                 #
# ------------------------------------------------------------------ #
def run_tianshou(n_steps: int = 4000) -> Dict[str, Any]:
    timer = defaultdict(int)                       # infer_ns, infer_steps

    train_envs = DummyVectorEnv([lambda: StepTimer(gym.make("LunarLander-v3"))])
    test_envs  = DummyVectorEnv([lambda: StepTimer(gym.make("LunarLander-v3"))])

    policy = make_policy(obs_shape=8, n_act=4, timer=timer)

    buf  = VectorReplayBuffer(total_size=20000, buffer_num=len(train_envs))
    c_tr = Collector(policy, train_envs, buf)
    c_te = Collector(policy, test_envs)

    trainer = OnpolicyTrainer(
        policy          = policy,
        train_collector = c_tr,
        test_collector  = c_te,
        max_epoch       = 1,
        step_per_epoch  = n_steps,
        step_per_collect = n_steps,      # rollout n_steps once
        repeat_per_collect = 1, batch_size = 64,
        episode_per_test   = 1, verbose=False,
    )

    t0 = time.perf_counter_ns()
    trainer.run()
    total_ns = time.perf_counter_ns() - t0

    # --------------------- 指标 ---------------------
    workers = list(train_envs.workers) + list(test_envs.workers)
    env_ns  = sum(w.env.env_time_ns for w in workers)
    steps   = sum(w.env.env_steps   for w in workers)
    infer_ns   = timer["infer_ns"]
    infer_steps = timer["infer_steps"] or 1          # 防止除零

    steps_s  = steps / (total_ns / 1e9)              # FPS
    env_ms   = env_ns   / 1e6 / steps
    infer_ms = infer_ns / 1e6 / infer_steps
    total_ms = total_ns / 1e6 / steps
    over_ms  = max(0.0, total_ms - env_ms - infer_ms)

    return {"steps/s": round(steps_s, 1),
            "env_ms":  round(env_ms,   3),
            "infer_ms":round(infer_ms, 3),
            "over_ms": round(over_ms,  3)}


# ------------------------------------------------------------------ #
# 4. Main: run 5 次取平均                                             #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    runs = [run_tianshou() for _ in range(5)]
    for i, r in enumerate(runs):
        print(f"Run {i}: {r}")
    avg = {k: round(float(np.mean([r[k] for r in runs])), 3) for k in runs[0]}
    print("\nAverage:", avg)
