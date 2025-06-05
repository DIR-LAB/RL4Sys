#!/usr/bin/env python
"""
Google Acme baseline for LunarLander-v3 with detailed timing.
"""

import time, gymnasium as gym, jax, jax.numpy as jnp
import haiku as hk
import acme
from acme import specs
from acme.wrappers import gym_wrapper
from acme.agents.agent import Agent
from acme.utils import tree_utils
from functools import partial
from collections import defaultdict
from numpy.random import default_rng

# -----------------------------------------------------------------------------
# 1. Policy network (2×64) – Haiku + JAX
# -----------------------------------------------------------------------------
def _mlp(obs: jnp.ndarray, num_actions: int) -> jnp.ndarray:
    mlp = hk.Sequential([
        hk.Linear(64), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(num_actions),
    ])
    return mlp(obs)

class AcmePolicy(Agent):
    """Stateless ε-greedy actor core for inference-only baseline."""
    def __init__(self, obs_spec, act_spec, rng_seed=0, epsilon=0.05):
        super().__init__()
        self._rng = jax.random.PRNGKey(rng_seed)
        self._epsilon = epsilon
        self._num_actions = act_spec.num_values

        def forward(obs):
            logits = _mlp(obs, self._num_actions)
            return logits
        self._net = hk.without_apply_rng(hk.transform(forward))

        # Initialise params once with dummy obs
        dummy = jnp.zeros(obs_spec.shape, dtype=jnp.float32)
        self._params = self._net.init(self._rng, dummy)

    # --- Agent API -----------------------------------------------------------
    def select_action(self, observation):
        # JAX forward pass
        logits = self._net.apply(self._params, observation.astype(jnp.float32))
        if jax.random.uniform(self._rng) < self._epsilon:
            # ε-greedy random
            return default_rng().integers(self._num_actions)
        else:
            return int(jnp.argmax(logits))

    def observe_first(self, timestep):   pass
    def observe(self, action, next_timestep):   pass
    def update(self):    pass  # No training

# -----------------------------------------------------------------------------
# 2. Profiling loop
# -----------------------------------------------------------------------------
def run_acme_baseline(num_steps: int = 4000):
    # -- wrap Gym env into Acme spec -----------------------------------------
    env = gym_wrapper.GymWrapper(gym.make("LunarLander-v3"))
    obs_spec, act_spec = specs.make_environment_spec(env).observations, env.action_spec()

    agent = AcmePolicy(obs_spec, act_spec)

    timers_ns = defaultdict(int)  # env, infer, total
    timestep = env.reset()

    for _ in range(num_steps):
        step_start = time.perf_counter_ns()

        # Inference -----------------------------------------------------------
        t0 = time.perf_counter_ns()
        action = agent.select_action(timestep.observation)
        timers_ns['infer'] += time.perf_counter_ns() - t0

        # Env step -----------------------------------------------------------
        t0 = time.perf_counter_ns()
        timestep = env.step(action)
        timers_ns['env'] += time.perf_counter_ns() - t0

        timers_ns['total'] += time.perf_counter_ns() - step_start

        if timestep.last():
            timestep = env.reset()

    # -- metrics -------------------------------------------------------------
    env_ms   = timers_ns['env']   / 1e6 / num_steps
    infer_ms = timers_ns['infer'] / 1e6 / num_steps
    total_ms = timers_ns['total'] / 1e6 / num_steps
    over_ms  = total_ms - env_ms - infer_ms

    print({
        "steps/s"  : round(1000 / total_ms, 1),
        "env_ms"   : round(env_ms,   3),
        "infer_ms" : round(infer_ms, 3),
        "over_ms"  : round(over_ms,  3),
    })

# -----------------------------------------------------------------------------
# 3. Run 5 rounds for stability
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    for i in range(5):
        run_acme_baseline()
