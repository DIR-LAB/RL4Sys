#!/usr/bin/env python
"""
Direct RL Baselines3 Zoo training script for profiling LunarLander performance.
This script uses the RL Zoo framework directly, similar to the command line interface.
"""
import sys
import os
import time
import types
import argparse
import numpy as np
from typing import Dict, Any, Optional, Callable
import gymnasium as gym

# Import RL Zoo components
try:
    from rl_zoo3.train import train
    from rl_zoo3 import ALGOS
    from rl_zoo3.utils import get_saved_hyperparams, create_test_env
    from rl_zoo3.exp_manager import ExperimentManager
except ImportError:
    print("RL Baselines3 Zoo not installed. Install with: pip install rl_zoo3")
    sys.exit(1)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import torch

# Add current directory to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env import StepTimer


class ProfilingCallback(BaseCallback):
    """Callback for profiling training performance during rollouts."""
    
    def __init__(self, verbose: int = 0):
        """Initialize the profiling callback.
        
        Args:
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.reset_timers()
    
    def reset_timers(self) -> None:
        """Reset all timing counters."""
        self.env_time = 0.0
        self.infer_time = 0.0
        self.total_time = 0.0
        self.step_count = 0
        self.rollout_start_time = None
    
    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.reset_timers()
        
        # Monkey patch the model's predict method for timing
        orig_forward = self.model.policy.forward        # <− 真实推理入口

        def timed_forward(self_pol, obs, deterministic=False):
            t0 = time.perf_counter_ns()
            action, value, log_prob = orig_forward(obs, deterministic)
            self.infer_time += time.perf_counter_ns() - t0
            return action, value, log_prob

        self.model.policy.forward = types.MethodType(timed_forward, self.model.policy)
        
        # Monkey patch the environment step method for timing
        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                if hasattr(env, 'env') and hasattr(env.env, 'step'):
                    self._patch_env_step(env.env)
    
    def _patch_env_step(self, env: gym.Env) -> None:
        """Patch environment step method to measure timing.
        
        Args:
            env: Environment to patch.
        """
        original_step = env.step
        
        def timed_step(action):
            t0 = time.perf_counter()
            result = original_step(action)
            self.env_time += time.perf_counter() - t0
            return result
        
        env.step = timed_step
    
    def _on_rollout_start(self) -> None:
        """Called at the start of each rollout."""
        self.rollout_start_time = time.perf_counter()
    
    def _on_step(self) -> bool:
        """Called after each environment step.
        
        Returns:
            bool: Whether to continue training.
        """
        self.step_count += 1
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        if self.rollout_start_time is not None:
            self.total_time += time.perf_counter() - self.rollout_start_time
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics.
        
        Returns:
            Dict[str, float]: Performance metrics.
        """
        if self.step_count == 0:
            return {"steps/s": 0, "env_ms": 0, "infer_ms": 0, "over_ms": 0}
        
        total_ms = (self.total_time * 1000) / self.step_count
        env_ms = (self.env_time * 1000) / self.step_count
        # Convert nanoseconds to milliseconds and divide by 2 since we infer twice
        infer_ms = (self.infer_time / 1e6) / self.step_count
        over_ms = total_ms - env_ms - infer_ms 
        
        return {
            "steps/s": round(1000 / total_ms, 1) if total_ms > 0 else 0,
            "env_ms": round(env_ms, 3),
            "infer_ms": round(infer_ms, 3),
            "over_ms": round(over_ms, 3),
        }


class SB3ZooTrainer:
    """Trainer using RL Baselines3 Zoo framework for profiling."""
    
    def __init__(self, algo: str = "ppo", env_id: str = "LunarLander-v3", 
                 num_timesteps: int = 4000):
        """Initialize the trainer.
        
        Args:
            algo: Algorithm to use (e.g., "ppo").
            env_id: Environment ID.
            num_timesteps: Number of timesteps to train.
        """
        self.algo = algo
        self.env_id = env_id
        self.num_timesteps = num_timesteps
        self.callback = ProfilingCallback()
    
    def create_env_wrapper(self) -> Callable:
        """Create environment wrapper factory.
        
        Returns:
            Callable: Environment factory function.
        """

        env = gym.make(self.env_id)
        return StepTimer(env)

    
    def train_and_profile(self) -> Dict[str, float]:
        """Train model and return profiling results.
        
        Returns:
            Dict[str, float]: Performance metrics.
        """
        policy_kwargs = {
            "net_arch": [64, 64],
            "activation_fn": torch.nn.ReLU,
        }
        env = self.create_env_wrapper()
        try:
            model = PPO(
                        "MlpPolicy",
                        env,    
                        learning_rate=3e-4,
                        n_steps=2048,
                        batch_size=64,
                        n_epochs=10,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=0.2,
                        clip_range_vf=None,
                        normalize_advantage=True,
                        ent_coef=0.0,
                        vf_coef=0.5,
                        max_grad_norm=0.5,
                        policy_kwargs=policy_kwargs,
                        verbose=0,
                        seed=0,
                        device="cpu",
                        tensorboard_log=None
                    )
        
            
            # Add our profiling callback
            self.callback.reset_timers()
            
            # Train the model
            start_time = time.perf_counter()
            model.learn(
                total_timesteps=self.num_timesteps,
                callback=self.callback,
                reset_num_timesteps=True,
                progress_bar=False
            )
            total_training_time = time.perf_counter() - start_time
            
            # Get performance metrics
            metrics = self.callback.get_performance_metrics()
            
            # Add total training time info
            metrics["total_training_time"] = round(total_training_time, 3)
            
            return metrics
            
        except Exception as e:
            print(f"Error during training: {e}")
            # Fallback to simple profiling

def main():
    """Main function to run SB3 Zoo training and profiling."""
    print("=" * 60)
    print("RL Baselines3 Zoo Training & Profiling")
    print("=" * 60)
    
    trainer = SB3ZooTrainer(
        algo="ppo",
        env_id="LunarLander-v3",
        num_timesteps=4000
    )
    
    # Run multiple profiling sessions
    results = []
    num_runs = 5
    
    for i in range(num_runs):
        print(f"\nRun {i + 1}/{num_runs}:")
        result = trainer.train_and_profile()
        result["run"] = i
        results.append(result)
        print(f"Results: {result}")
    
    # Calculate averages
    if results:
        avg_result = {}
        for key in ["steps/s", "env_ms", "infer_ms", "over_ms"]:
            if key in results[0]:
                avg_result[key] = round(
                    np.mean([r[key] for r in results]), 3
                )
        
        print("\n" + "=" * 60)
        print(f"Average Results: {avg_result}")
        print("=" * 60)


if __name__ == "__main__":
    main() 