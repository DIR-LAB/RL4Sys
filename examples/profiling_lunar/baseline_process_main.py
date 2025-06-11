#!/usr/bin/env python
"""
Baseline vs. Multiprocess‑queue trajectory collection benchmark
---------------------------------------------------------------

 * ``run_baseline`` – unchanged single‑thread reference.
 * ``run_mproc``   – producer runs env + policy in the **main process** and
   streams each transition to a background **consumer process** via
   ``multiprocessing.Queue``.  Measures the extra cost of IPC
   (``queue_ms``).

macOS defaults to the *spawn* start‑method, so we guard all multiprocessing
code under ``if __name__ == "__main__"`` and set the start‑method
explicitly.

Run the file – it prints five baseline runs, then five multi‑process runs
for easy comparison.
"""
import time
import multiprocessing as mp
import queue  # only for Empty exception in consumer
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

torch.set_num_threads(1)

# ‑‑‑‑‑‑‑ Timed environment wrapper ‑‑‑‑‑‑‑------------------------------------------------
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

# ‑‑‑‑‑‑‑ Simple MLP policy ‑‑‑‑‑‑‑--------------------------------------------------------
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
    def act(self, obs: torch.Tensor) -> int:
        logits = self.net(obs)
        return Categorical(logits=logits).sample().item()

# ‑‑‑‑‑‑‑ Baseline (no queue) ‑‑‑‑‑‑‑------------------------------------------------------

def run_baseline(num_steps: int = 4000):
    env = TimedEnv(None)
    policy = SimplePolicy(env.observation_space.shape[0], env.action_space.n)

    obs, _ = env.reset(seed=0)
    env_time = infer_time = 0.0
    t0_total = time.perf_counter()

    for _ in range(num_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        t0 = time.perf_counter()
        action = policy.act(obs_t)
        infer_time += time.perf_counter() - t0

        obs, _, done, trunc, info = env.step(action)
        env_time += info["step_time"]

        if done or trunc:
            obs, _ = env.reset()

    per_step_ms = (time.perf_counter() - t0_total) * 1000 / num_steps
    env_ms  = env_time  * 1000 / num_steps
    infer_ms = infer_time * 1000 / num_steps
    over_ms  = per_step_ms - env_ms - infer_ms

    print({
        "steps/s": round(1000 / per_step_ms, 1),
        "env_ms":  round(env_ms,   3),
        "infer_ms":round(infer_ms, 3),
        "over_ms": round(over_ms,  3),
    })

# ‑‑‑‑‑‑‑ Multiprocess variant ‑‑‑‑‑‑‑-----------------------------------------------------

def consumer(q: mp.Queue):
    STOP = None  # sentinel must be picklable
    while True:
        item = q.get()
        if item is STOP:
            break
        # simulate storage/learning latency here if desired
    q.close()

def run_mproc(num_steps: int = 4000, q_maxsize: int = 10000):
    """Producer‑consumer using ``multiprocessing.Queue``."""
    env = TimedEnv(None)
    policy = SimplePolicy(env.observation_space.shape[0], env.action_space.n)

    mp_queue: "mp.Queue[Tuple]" = mp.Queue(maxsize=q_maxsize)
    STOP = None  # sentinel must be picklable

    proc = mp.Process(target=consumer, args=(mp_queue,), daemon=True)
    proc.start()

    obs, _ = env.reset(seed=0)
    env_time = infer_time = q_time = 0.0
    t0_total = time.perf_counter()

    for _ in range(num_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        # inference
        t0 = time.perf_counter()
        action = policy.act(obs_t)
        infer_time += time.perf_counter() - t0

        # env step
        obs_next, reward, done, trunc, info = env.step(action)
        env_time += info["step_time"]

        # enqueue (IPC)
        t0 = time.perf_counter()
        mp_queue.put((obs, action, reward, obs_next, done or trunc))
        q_time += time.perf_counter() - t0

        obs = obs_next if not (done or trunc) else env.reset()[0]

    per_step_ms = (time.perf_counter() - t0_total) * 1000 / num_steps
    env_ms   = env_time   * 1000 / num_steps
    infer_ms = infer_time * 1000 / num_steps
    queue_ms = q_time     * 1000 / num_steps
    over_ms  = per_step_ms - env_ms - infer_ms - queue_ms

    print({
        "steps/s":    round(1000 / per_step_ms, 1),
        "env_ms":     round(env_ms,   3),
        "infer_ms":   round(infer_ms, 3),
        "queue_ms":   round(queue_ms, 3),
        "over_ms":    round(over_ms,  3),
    })

    mp_queue.put(STOP)  # signal consumer to exit
    proc.join()
    mp_queue.close()

# ‑‑‑‑‑‑‑ Entry point ‑‑‑‑‑‑‑--------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # portable across Mac/Win/Linux
    torch.set_num_threads(1)                   # avoid extra thread contention

    print("Baseline runs:")
    for _ in range(5):
        run_baseline()

    print("\nMultiprocess‑queue runs:")
    for _ in range(5):
        run_mproc()
