import argparse
import time
import random
import numpy as np
import torch

from pathlib import Path

# Add project root to path so imports work when run directly
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from rl4sys.client.agent import RL4SysAgent

class SimpleEnv:
    """A minimal stub environment that produces random observations and rewards."""
    def __init__(self, input_dim: int, act_dim: int, seed: int = 0):
        self.input_dim = input_dim
        self.act_dim = act_dim
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return self._sample_observation()

    def step(self, action: int, max_steps: int):
        # Action is ignored in this stub – replace with real logic for a genuine env.
        self.step_count += 1
        next_obs = self._sample_observation()
        reward = float(self.rng.uniform(-1.0, 1.0))
        done = self.step_count >= max_steps
        return next_obs, reward, done

    def _sample_observation(self):
        return self.rng.uniform(-1.0, 1.0, size=self.input_dim).astype(np.float32)


def build_observation(obs: np.ndarray) -> torch.Tensor:
    """Convert a numpy observation into the tensor shape expected by the agent."""
    return torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="RL4Sys Python Test Simulation")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--number-of-iterations', type=int, default=100)
    parser.add_argument('--number-of-moves', type=int, default=500)
    parser.add_argument('--config', type=str, default=str(Path(__file__).with_name('test_conf.json')))

    args = parser.parse_args()

    print(f"RL4Sys Python Test Simulation")
    print(f"Seed: {args.seed}, Iterations: {args.number_of_iterations}, Max moves: {args.number_of_moves}")
    print(f"Using config: {args.config}")

    # Initialize random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        agent = RL4SysAgent(conf_path=args.config, debug=False)
        # Retrieve input and action dimensions from config if available
        input_dim = agent.algorithm_parameters.get('input_size', 8)
        act_dim = agent.algorithm_parameters.get('act_dim', 4)

        env = SimpleEnv(input_dim=input_dim, act_dim=act_dim, seed=args.seed)

        for iteration in range(args.number_of_iterations):
            print(f"\nIteration {iteration}")
            obs = env.reset()
            obs_tensor = build_observation(obs)
            traj = None  # Initially no trajectory
            cumulative_reward = 0.0
            moves = 0
            start_time = time.time()

            while moves < args.number_of_moves:
                # Ask the agent for an action
                try:
                    traj, rl_action = agent.request_for_action(traj, obs_tensor)
                except Exception as e:
                    # If the server is unreachable or another error occurs, fall back to random
                    print(f"RequestForAction failed: {e}. Falling back to random action.")
                    if traj is None or traj.is_completed():
                        traj = agent.request_for_action(traj=None, obs_tensor=obs_tensor)[0]
                    random_action_value = random.randint(0, act_dim - 1)
                    rl_action.update_reward(0.0)
                    rl_action.act = torch.tensor(random_action_value)
                    agent.add_to_trajectory(traj, rl_action)
                    next_obs, reward, done = env.step(random_action_value, args.number_of_moves)
                else:
                    # Execute the chosen action in the environment
                    action_value = int(rl_action.act.item()) if isinstance(rl_action.act, torch.Tensor) else int(rl_action.act)
                    next_obs, reward, done = env.step(action_value, args.number_of_moves)
                    rl_action.update_reward(reward)
                    agent.add_to_trajectory(traj, rl_action)

                cumulative_reward += reward
                moves += 1
                obs_tensor = build_observation(next_obs)

                if done:
                    agent.mark_end_of_trajectory(traj, rl_action)
                    break

            elapsed = time.time() - start_time
            print(f"Iteration completed. Moves: {moves}, Cumulative reward: {cumulative_reward:.3f}, Time(s): {elapsed:.2f}")

    except Exception as e:
        print(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    script_start_time = time.time()
    main()
    script_end_time = time.time()
    # time in seconds
    print(f"Script completed. Time(s): {script_end_time - script_start_time:.3f} seconds")
