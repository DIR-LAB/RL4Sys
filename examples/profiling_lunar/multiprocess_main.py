#!/usr/bin/env python
"""
CPU 亲和性干扰实验（完整版）
===========================

**目的**
--------
在保持原有 RL 采样循环（`env.step` + `policy.act`）不变的前提下，
通过额外启动一个 **CPU‑intensive 子进程**，测量它对主进程性能的影响。
脚本同时支持：

* **基线**（单进程，无任何干扰）。
* **干扰模式**：主、子进程可绑定到不同 CPU 核心；子进程提供两种压力源：
  * 纯 Python `sqrt` 累加（单线程，GIL 激烈）。
  * 大型矩阵乘 `1024×1024`（BLAS，多线程，高内存/带宽）。

运行方式
--------
```bash
# 1) 仅跑基线 + 干扰各 5 次，默认主核=0，压核=1，单线程 sqrt
python cpu_affinity_bench.py

# 2) 换成矩阵乘压力，并显式指定核心：
MAIN_CORE=2  STRESS_CORE=3  STRESS_MATMUL=1  python cpu_affinity_bench.py
```

> macOS 13+/Linux: 使用 `psutil` 或 `os.sched_setaffinity` 绑定核心。
> Windows 也可跑，但若无法设置亲和性则会回落为“未绑核”行为。
"""
import os
import time
import math
import multiprocessing as mp
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

# -----------------------------------------------------------------------------
# 亲和性工具 – Linux: sched_setaffinity ； 其余平台用 psutil
# -----------------------------------------------------------------------------
try:
    import psutil
    def pin_process(core_id: int):
        try:
            psutil.Process().cpu_affinity([core_id])
        except AttributeError:
            pass  # macOS < 13 无此接口，忽略
except ImportError:  # psutil 不可用 → 仅 Linux sched_setaffinity
    def pin_process(core_id: int):
        try:
            import os
            os.sched_setaffinity(0, {core_id})
        except (AttributeError, PermissionError, OSError):
            pass  # 无法绑定就算了

# -----------------------------------------------------------------------------
# 环境包装器 – 记录 env.step() 开销
# -----------------------------------------------------------------------------
class TimedEnv(gym.Env):
    def __init__(self, _):
        self.env = gym.make("LunarLander-v3")

    def reset(self, **kw):
        return self.env.reset(**kw)

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

# -----------------------------------------------------------------------------
# 两层 MLP 策略
# -----------------------------------------------------------------------------
class SimplePolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    @torch.inference_mode()
    def act(self, obs: torch.Tensor) -> int:
        logits = self.net(obs)
        return Categorical(logits=logits).sample().item()

# -----------------------------------------------------------------------------
# 基线 – 无任何干扰
# -----------------------------------------------------------------------------

def run_baseline(num_steps: int = 4000):
    env = TimedEnv(None)
    policy = SimplePolicy(env.observation_space.shape[0], env.action_space.n)

    obs, _ = env.reset(seed=0)
    env_t = infer_t = 0.0
    t0_tot = time.perf_counter()

    for _ in range(num_steps):
        t0 = time.perf_counter(); act = policy.act(torch.as_tensor(obs, dtype=torch.float32)); infer_t += time.perf_counter() - t0
        obs, _, d, tr, info = env.step(act); env_t += info["step_time"]
        if d or tr:
            obs, _ = env.reset()

    per = (time.perf_counter() - t0_tot) * 1e3 / num_steps
    print({
        "steps/s": round(1000/per, 1),
        "env_ms":  round(env_t*1e3/num_steps, 3),
        "infer_ms":round(infer_t*1e3/num_steps, 3),
        "over_ms": round(per - env_t*1e3/num_steps - infer_t*1e3/num_steps, 3),
    })

# -----------------------------------------------------------------------------
# CPU 压力进程 – 支持两种模式
# -----------------------------------------------------------------------------

def cpu_stressor(stop_evt: mp.Event, core_id: int):
    pin_process(core_id)
    import numpy as np

    if os.getenv("STRESS_MATMUL") == "1":
        # 大矩阵 GEMM，多线程 BLAS
        a = np.random.rand(1024, 1024).astype(np.float32)
        b = np.random.rand(1024, 1024).astype(np.float32)
        while not stop_evt.is_set():
            np.matmul(a, b, out=a)
    else:
        acc = 0.0
        while not stop_evt.is_set():
            for i in range(10_000):
                acc += math.sqrt(abs(math.sin(i)))
        if acc < 0:  # 防优化
            print(acc)

# -----------------------------------------------------------------------------
# 基线 + 压力进程
# -----------------------------------------------------------------------------

def run_with_stressor(num_steps: int = 4000,
                       main_core: int = 0,
                       stress_core: int = 1):

    pin_process(main_core)
    stop_evt = mp.Event()
    p = mp.Process(target=cpu_stressor, args=(stop_evt, stress_core), daemon=True)
    p.start()

    env = TimedEnv(None)
    policy = SimplePolicy(env.observation_space.shape[0], env.action_space.n)

    obs, _ = env.reset(seed=0)
    env_t = infer_t = 0.0
    t0_tot = time.perf_counter()

    for _ in range(num_steps):
        t0 = time.perf_counter(); act = policy.act(torch.as_tensor(obs, dtype=torch.float32)); infer_t += time.perf_counter() - t0
        obs, _, d, tr, info = env.step(act); env_t += info["step_time"]
        if d or tr:
            obs, _ = env.reset()

    per = (time.perf_counter() - t0_tot) * 1e3 / num_steps
    print({
        "steps/s": round(1000/per, 1),
        "env_ms":  round(env_t*1e3/num_steps, 3),
        "infer_ms":round(infer_t*1e3/num_steps, 3),
        "over_ms": round(per - env_t*1e3/num_steps - infer_t*1e3/num_steps, 3),
    })

    stop_evt.set(); p.join()

# -----------------------------------------------------------------------------
# 主入口 – 连续跑 5 次基线 & 干扰
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.set_num_threads(1)

    main_core   = int(os.getenv("MAIN_CORE", "0"))
    stress_core = int(os.getenv("STRESS_CORE", "1"))

    print("Baseline runs (core", main_core, "):")
    for _ in range(5):
        run_baseline()

    print("\nWith CPU‑stressor runs (main_core=", main_core,
          ", stress_core=", stress_core, ", matmul=", os.getenv("STRESS_MATMUL", "0"), "):")
    for _ in range(5):
        run_with_stressor(main_core=main_core, stress_core=stress_core)
