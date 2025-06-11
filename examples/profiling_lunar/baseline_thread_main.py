#!/usr/bin/env python
"""
Baseline vs. Threaded‑queue trajectory collection benchmark
-----------------------------------------------------------

 * ``run_baseline`` replicates the original single‑thread timing loop.
 * ``run_threaded`` pushes each (obs, action, reward, next_obs, done)
   tuple into a :pyclass:`queue.Queue` while a background consumer
   thread dequeues and discards the data (simulating an I/O pipeline or
   learner process).  The function measures the extra cost of the queue
   push (``queue_ms``) in addition to env‑step, inference and residual
   overhead.

Both functions print a dictionary so you can copy/paste results into a
spreadsheet for comparison.
"""
import time
import threading
import queue
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


# ---------------- Timed environment wrapper -----------------------------
class TimedEnv(gym.Env):
    def __init__(self, _):
        self.env = gym.make("LunarLander-v3")

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        t0 = time.perf_counter()
        obs, reward, done, trunc, info = self.env.step(action)
        info["step_time"] = time.perf_counter() - t0
        return obs, reward, done, trunc, info

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


# ---------------- two‑layer MLP policy ----------------------------------
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

    #@torch.no_grad()
    @torch.inference_mode()
    def act(self, obs: torch.Tensor) -> int:
        logits = self.net(obs)
        return Categorical(logits=logits).sample().item()

# ---------------- baseline (no queue) -----------------------------------

def run_baseline(num_steps: int = 4000):
    env = TimedEnv(None)
    policy = SimplePolicy(env.observation_space.shape[0], env.action_space.n)

    obs, _ = env.reset(seed=0)
    env_time = infer_time = 0.0
    t0_total = time.perf_counter()

    for _ in range(num_steps):
        obs_t = torch.from_numpy(obs)
        # 1️⃣  inference
        t0 = time.perf_counter()
        action = policy.act(obs_t)
        infer_time += time.perf_counter() - t0

        # 2️⃣  env step
        obs, _, done, trunc, info = env.step(action)
        env_time += info["step_time"]

        if done or trunc:
            obs, _ = env.reset()

    per_step_ms = (time.perf_counter() - t0_total) * 1000 / num_steps
    env_ms = env_time * 1000 / num_steps
    infer_ms = infer_time * 1000 / num_steps
    over_ms = per_step_ms - env_ms - infer_ms

    print({
        "steps/s": round(1000 / per_step_ms, 1),
        "env_ms": round(env_ms, 3),
        "infer_ms": round(infer_ms, 3),
        "over_ms": round(over_ms, 3),
    })

# ---------------- baseline (with queue) -----------------------------------

def run_baseline_with_queue(num_steps: int = 4000, q_maxsize: int = 10000):
    env = TimedEnv(None)
    policy = SimplePolicy(env.observation_space.shape[0], env.action_space.n)

    traj_q: "queue.Queue[Tuple]" = queue.Queue(maxsize=q_maxsize)

    obs, _ = env.reset(seed=0)
    env_time = infer_time = q_time = 0.0
    t0_total = time.perf_counter()

    for _ in range(num_steps):
        obs_t = torch.from_numpy(obs)
        # 1️⃣  inference
        t0 = time.perf_counter()
        action = policy.act(obs_t)
        infer_time += time.perf_counter() - t0

        # 2️⃣  env step
        obs, _, done, trunc, info = env.step(action)
        env_time += info["step_time"]

        # 3️⃣ enqueue transition
        t0 = time.perf_counter()
        traj_q.put((obs, action, 0, obs, done or trunc))
        q_time += time.perf_counter() - t0

        if done or trunc:
            obs, _ = env.reset()

    per_step_ms = (time.perf_counter() - t0_total) * 1000 / num_steps
    env_ms = env_time * 1000 / num_steps
    infer_ms = infer_time * 1000 / num_steps
    queue_ms = q_time * 1000 / num_steps
    over_ms = per_step_ms - env_ms - infer_ms - queue_ms

    print({
        "steps/s": round(1000 / per_step_ms, 1),
        "env_ms": round(env_ms, 3),
        "infer_ms": round(infer_ms, 3),
        "queue_ms": round(queue_ms, 3),
        "over_ms": round(over_ms, 3),
    })


# ---------------- threaded‑queue variant ---------------------------------

def run_threaded(num_steps: int = 4000, q_maxsize: int = 10000):
    """Same loop but pushes each transition into a Queue processed by a
    background thread.  Reports extra queue push cost (queue_ms)."""
    env = TimedEnv(None)
    policy = SimplePolicy(env.observation_space.shape[0], env.action_space.n)

    traj_q: "queue.Queue[Tuple]" = queue.Queue(maxsize=q_maxsize)
    stop_token = object()  # sentinel to shut the consumer down

    def consumer():
        while True:
            item = traj_q.get()
            if item is stop_token:
                traj_q.task_done()
                break  # graceful exit
            # here you'd write to replay buffer / disk / whatever
            traj_q.task_done()

    t_consumer = threading.Thread(target=consumer, daemon=True)
    t_consumer.start()

    obs, _ = env.reset(seed=0)
    env_time = infer_time = q_time = 0.0
    t0_total = time.perf_counter()

    for _ in range(num_steps):
        obs_t = torch.from_numpy(obs)
        # 1️⃣ inference
        t0 = time.perf_counter()
        action = policy.act(obs_t)
        infer_time += time.perf_counter() - t0

        # 2️⃣ env step
        obs_next, reward, done, trunc, info = env.step(action)
        env_time += info["step_time"]

        # 3️⃣ enqueue transition
        t0 = time.perf_counter()
        traj_q.put((obs, action, reward, obs_next, done or trunc))
        q_time += time.perf_counter() - t0

        obs = obs_next if not (done or trunc) else env.reset()[0]

        if done or trunc:
            obs, _ = env.reset()

    per_step_ms = (time.perf_counter() - t0_total) * 1000 / num_steps
    env_ms = env_time * 1000 / num_steps
    infer_ms = infer_time * 1000 / num_steps
    queue_ms = q_time * 1000 / num_steps
    over_ms = per_step_ms - env_ms - infer_ms - queue_ms

    print({
        "steps/s": round(1000 / per_step_ms, 1),
        "env_ms": round(env_ms, 3),
        "infer_ms": round(infer_ms, 3),
        "queue_ms": round(queue_ms, 3),
        "over_ms": round(over_ms, 3),
    })

    # 4️⃣ shut down consumer and wait until it drains the queue
    traj_q.put(stop_token)
    traj_q.join()  # ensures consumer processed sentinel
    t_consumer.join()


if __name__ == "__main__":

    print("Baseline runs:")
    for _ in range(5):
        run_baseline()

    print("\nBaseline with queue runs:")
    for _ in range(5):
        run_baseline_with_queue()

    print("\nThreaded‑queue runs:")
    for _ in range(5):
        run_threaded()
