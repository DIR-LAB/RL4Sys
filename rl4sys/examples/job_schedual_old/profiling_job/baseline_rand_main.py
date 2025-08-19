#!/usr/bin/env python
"""Baseline performance benchmark for the HPCSim job-scheduling environment.

This script replicates the timing study originally performed on
``LunarLander-v3`` but for the HPCSim job-scheduling task.  It measures the
average wall-clock time spent in

1. neural-network inference using the *kernel* architecture from
   ``rl4sys.algorithms.PPO.kernel`` and
2. the environment transition performed by :class:`HPCEnv`.

Example
-------
Run the benchmark for 4 k environment steps and print timing statistics::

    python examples/profiling_job/baseline_main.py

The script executes 20 independent runs (identical to the original baseline)
and reports a summary dictionary per run:
``{"steps/s": 1234.5, "env_ms": 0.321, "infer_ms": 0.045, "over_ms": 0.012}``.

Notes
-----
* Only **inference** is performed – the policy network parameters are
  *randomly initialised* and **never** updated.
* The action-selection logic applies the same masking rules as used during
  PPO training to avoid sampling invalid actions.
* All computations run on CPU; no GPU is required.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Tuple

import numpy as np
import torch
import random

# ---------------------------------------------------------------------------
# Dynamically add *HPCSim* directory so its intra-module imports resolve.
# ---------------------------------------------------------------------------
THIS_DIR: str = os.path.dirname(__file__)

# -----------------------------------------------------------------------
# Directory layout (relative to *THIS_DIR*):
#   ..               -> rl4sys/examples/job_schedual_old
#   ../HPCSim        -> HPCSim package containing job.py, cluster.py, etc.
# -----------------------------------------------------------------------

# Path to rl4sys/examples/job_schedual_old (parent directory of HPCSim)
HPCSIM_PARENT_DIR: str = os.path.abspath(os.path.join(THIS_DIR, ".."))
# Path to the HPCSim package itself
HPCSIM_DIR: str = os.path.join(HPCSIM_PARENT_DIR, "HPCSim")

# Append both directories to *sys.path* so that:
#   • "import HPCSim.HPCSimPickJobs" resolves as a *package* import, and
#   • top-level imports inside HPCSim modules (e.g. "from job import Job")
#     are also found because *HPCSIM_DIR* contains job.py.
sys.path.insert(0, HPCSIM_PARENT_DIR)
sys.path.insert(0, HPCSIM_DIR)

# ---------------------------------------------------------------------------
# Ensure the *RL4Sys* project root is on ``sys.path`` so we can import
# ``rl4sys.algorithms`` when this script is executed directly.
# ---------------------------------------------------------------------------
PROJECT_ROOT: str = os.path.abspath(os.path.join(THIS_DIR, "../../../../"))
sys.path.insert(0, PROJECT_ROOT)

# pylint: disable=wrong-import-position
from HPCSim.HPCSimPickJobs import HPCEnv  # type: ignore
# Network inference removed for random baseline; RL4Sys imports no longer needed.
from rl4sys.utils.mem_prof import MemoryProfiler

# Environment constants (keep in sync with HPCSim implementation)
JOB_FEATURES: int = 8
MAX_QUEUE_SIZE: int = 128
OBS_DIM: int = JOB_FEATURES * MAX_QUEUE_SIZE  # 1024
ACT_DIM: int = MAX_QUEUE_SIZE  # 128


class TimedEnv:
    """Wrapper that records the duration of each :py:meth:`env.step`."""

    def __init__(self, env: HPCEnv) -> None:
        self.env: HPCEnv = env

    def reset(self, **kwargs):  # noqa: D401, D403 – pass-through wrapper
        """Delegate to the underlying :py:meth:`HPCEnv.reset`."""
        return self.env.reset(**kwargs)

    def step(self, action: int):
        """Time‐instrumented call to :py:meth:`HPCEnv.step`."""
        t0 = time.perf_counter()
        result = self.env.step(action)
        step_time = time.perf_counter() - t0
        # Preserve original return structure and append timing information.
        if isinstance(result, list):
            result.append({"step_time": step_time})
        return result


# ---------------------------------------------------------------------------
# No neural network is used in this *random* baseline.  Actions are sampled
# uniformly from the set of *valid* indices according to ``build_mask``.
# ---------------------------------------------------------------------------


def build_mask(obs: np.ndarray) -> np.ndarray:
    """Construct an action-validity mask identical to PPO training logic.

    A slot is *invalid* (mask = 0) if its 8-dimensional feature vector equals
    either of two special patterns used in HPCSim:
    1. Empty slot  – ``[0, 1, 1, 1, 1, 1, 1, 0]``
    2. Filled slot – ``[1, 1, 1, 1, 1, 1, 1, 1]``
    Any other pattern corresponds to a job that can be scheduled and thus
    receives mask = 1.
    """
    mask = np.ones(ACT_DIM, dtype=np.float32)
    empty = np.array([0] + [1] * (JOB_FEATURES - 2) + [0])
    full = np.ones(JOB_FEATURES, dtype=np.float32)

    for idx in range(0, OBS_DIM, JOB_FEATURES):
        slot = obs[idx : idx + JOB_FEATURES]
        if np.allclose(slot, empty) or np.allclose(slot, full):
            mask[idx // JOB_FEATURES] = 0.0
    return mask


def run_baseline(env = None, num_steps: int = 4000, workload_file: str | None = None) -> None:
    """Execute *num_steps* interactions and report timing statistics."""

    # ---------------------------------------------------------------------
    # Environment initialisation
    # ---------------------------------------------------------------------

    

    timed_env = TimedEnv(env)

    # ---------------------------------------------------------------------
    # In this baseline, inference is replaced by uniform random sampling over
    # valid actions.  We still measure the time spent selecting an action to
    # maintain the same timing breakdown (``infer_time`` now represents the
    # sampling overhead).
    # ---------------------------------------------------------------------

    obs, _ = timed_env.reset()
    env_time = 0.0
    infer_time = 0.0
    t0_total = time.perf_counter()

    for _ in range(num_steps):
        mask_np = build_mask(obs)

        # ------------------------ Action sampling -----------------------
        t0 = time.perf_counter()
        valid_indices = np.flatnonzero(mask_np)
        if valid_indices.size == 0:
            action = 0  # fallback to 0 if no valid action (should not happen)
        else:
            action = int(random.choice(valid_indices))
        infer_time += time.perf_counter() - t0

        # --------------------- Environment transition --------------------
        step_result = timed_env.step(action)
        step_time_dict = step_result[-1]  # appended by TimedEnv
        env_time += step_time_dict["step_time"]

        done = step_result[2]
        obs = step_result[0] if not done else timed_env.reset()[0]

    # ---------------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------------
    total_ms = (time.perf_counter() - t0_total) * 1000.0
    per_step_ms = total_ms / num_steps
    env_ms = env_time * 1000.0 / num_steps
    infer_ms = infer_time * 1000.0 / num_steps
    over_ms = per_step_ms - env_ms - infer_ms

    print(
        {
            "steps/s": round(1000.0 / per_step_ms, 1),
            "env_ms": round(env_ms, 3),
            "infer_ms": round(infer_ms, 3),
            "over_ms": round(over_ms, 3),
        }
    )


if __name__ == "__main__":
    env = HPCEnv()
    env.seed(0)
    env.my_init(workload_file=os.path.join(HPCSIM_DIR, "data", "lublin_256.swf"), sched_file="")
    

    profiler = MemoryProfiler(proj_name="job_scheduling_baseline", output_dir="memtest")
    profiler.start_background_profiling()

    for _ in range(20):
        run_baseline(env)

    profiler.stop_background_profiling()