import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../rl4sys/proto')))

import numpy as np
import random
import time
import torch
import gymnasium as gym
import gymnasium.spaces
from typing import Tuple, Dict

from rl4sys.client.agent import RL4SysAgent
from rl4sys.utils.util import StructuredLogger
from rl4sys.utils.logging_config import setup_rl4sys_logging
from torch.utils.tensorboard import SummaryWriter


"""
Environment script: Lunar Lander Continuous Simulator

Training server parameters:
    Uses PPO_Continuous (Gaussian policy) via RL4Sys server
"""


class LunarLanderContinuousSim():
    def __init__(self, seed: int, client_id: str, render_game: bool = False, debug: bool = False):
        self._seed = seed
        self._client_id = client_id
        self._render_game = render_game
        self.logger = StructuredLogger("LunarLanderContinuousSim", debug=debug)
        self.tensorboard_writer = SummaryWriter(log_dir='./logs/lunar_lander_continuous')

        # Initialize the Gym environment (continuous control)
        self.env = gym.make('LunarLander-v3', continuous=True)

        # Set the seeds for reproducibility
        self.env.reset(seed=self._seed)
        self.env.action_space.seed(self._seed)
        self.env.observation_space.seed(self._seed)
        np.random.seed(self._seed)
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        assert isinstance(self.env.action_space, gym.spaces.Box), "Expected Box action space for continuous LunarLander"
        self.act_dim = int(self.env.action_space.shape[0])
        self.act_low = self.env.action_space.low
        self.act_high = self.env.action_space.high

        # Continuous PPO agent configuration
        self.rlagent = RL4SysAgent(conf_path='./rl4sys/examples/lunar/luna_conf_cont.json')
        self.rl4sys_traj = None
        self.rl4sys_action = None

        # To store simulator stats
        self.simulator_stats: Dict[str, object] = {
            'moves': 0,
            'action_rewards': [],
            'success_count': 0,
            'death_count': 0,
            'time_to_goal': [],
            'time_to_death': [],
            'total_iterations': 0
        }

        self.logger.info(
            "Initialized LunarLanderContinuousSim",
            client_id=self._client_id,
            seed=self._seed,
            render_game=self._render_game,
            action_space_type="continuous",
            action_dim=self.act_dim,
        )

    def run_application(self, num_iterations: int, max_moves: int) -> None:
        self.logger.info(
            "Starting simulation",
            num_iterations=num_iterations,
            max_moves=max_moves
        )

        env_time_acc: float = 0.0
        infer_time_acc: float = 0.0
        total_step_count: int = 0
        t0_total: float = time.perf_counter()

        for iteration in range(num_iterations):
            self.simulator_stats['total_iterations'] += 1
            self.logger.info(
                "Starting iteration",
                iteration=iteration,
                total_iterations=self.simulator_stats['total_iterations']
            )

            obs, _ = self.env.reset(seed=self._seed + iteration)
            done = False
            moves = 0
            cumulative_reward = 0.0
            start_time = time.time()

            obs_tensor = self.build_observation(obs)

            env_ns = 0
            infer_ns = 0

            while not done and moves < max_moves:
                if self._render_game:
                    self.env.render()

                t0 = time.perf_counter_ns()
                self.rl4sys_traj, self.rl4sys_action = self.rlagent.request_for_action(self.rl4sys_traj, obs_tensor)
                infer_ns += time.perf_counter_ns() - t0

                # Append to trajectory immediately
                self.rlagent.add_to_trajectory(self.rl4sys_traj, self.rl4sys_action)

                # Convert action to numpy vector within bounds
                action = self.rl4sys_action.act
                if isinstance(action, torch.Tensor):
                    action = action.detach().cpu().numpy()
                action = np.asarray(action, dtype=np.float32).reshape(self.act_dim)
                action = np.clip(action, self.act_low, self.act_high)

                t0 = time.perf_counter_ns()
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                env_ns += time.perf_counter_ns() - t0

                done = bool(terminated or truncated)
                cumulative_reward += float(reward)

                next_obs_tensor = self.build_observation(next_obs)
                self.rl4sys_action.update_reward(float(reward))

                moves += 1
                self.simulator_stats['moves'] += 1
                self.simulator_stats['action_rewards'].append(float(reward))

                obs_tensor = next_obs_tensor

                if moves >= max_moves or done:
                    self.rl4sys_action.done = True
                    self.rlagent.mark_end_of_trajectory(self.rl4sys_traj, self.rl4sys_action)

                    if cumulative_reward >= 200:
                        self.simulator_stats['success_count'] += 1
                        self.simulator_stats['time_to_goal'].append(time.time() - start_time)
                        self.logger.info(
                            "Successful landing",
                            iteration=iteration,
                            time_to_goal=time.time() - start_time,
                            success_count=self.simulator_stats['success_count']
                        )
                    else:
                        self.simulator_stats['death_count'] += 1
                        self.simulator_stats['time_to_death'].append(time.time() - start_time)
                        self.logger.info(
                            "Failed landing",
                            iteration=iteration,
                            time_to_death=time.time() - start_time,
                            death_count=self.simulator_stats['death_count']
                        )
                    break

            # Log reward per episode
            self.tensorboard_writer.add_scalar('reward', cumulative_reward, iteration)

            # Accumulate profiling in seconds
            env_time_acc += env_ns * 1e-9
            infer_time_acc += infer_ns * 1e-9
            total_step_count += moves

        # Final aggregated performance metrics
        total_elapsed = time.perf_counter() - t0_total
        if total_step_count > 0:
            per_step_ms = (total_elapsed * 1000.0) / total_step_count
            env_ms = (env_time_acc * 1000.0) / total_step_count
            infer_ms = (infer_time_acc * 1000.0) / total_step_count
            over_ms = per_step_ms - env_ms - infer_ms

            performance_summary = {
                "steps/s": round(1000.0 / per_step_ms, 1),
                "env_ms": round(env_ms, 3),
                "infer_ms": round(infer_ms, 3),
                "over_ms": round(over_ms, 3),
            }
            print("Performance summary:", performance_summary)

        if self._render_game:
            self.env.close()

        self.logger.info(
            "Simulation completed",
            total_iterations=self.simulator_stats['total_iterations'],
            total_moves=self.simulator_stats['moves'],
            success_count=self.simulator_stats['success_count'],
            death_count=self.simulator_stats['death_count'],
            avg_time_to_goal=np.mean(self.simulator_stats['time_to_goal']) if self.simulator_stats['time_to_goal'] else 0,
            avg_time_to_death=np.mean(self.simulator_stats['time_to_death']) if self.simulator_stats['time_to_death'] else 0,
            avg_reward=np.mean(self.simulator_stats['action_rewards']) if self.simulator_stats['action_rewards'] else 0,
        )

    def build_observation(self, obs):
        return torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog="RL4Sys Lunar Lander Continuous Simulator",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random number generation in environment')
    parser.add_argument('--number-of-iterations', type=int, default=20000,
                        help='number of iterations to run')
    parser.add_argument('--number-of-moves', type=int, default=400,
                        help='maximum number of moves allowed per iteration')
    parser.add_argument('--render', type=bool, default=False,
                        help='render the Lunar Lander environment')
    parser.add_argument('--client-id', type=str, default="lunar_lander_continuous",
                        help='unique client id for the simulation')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args, extras = parser.parse_known_args()

    # Initialize logging system
    from rl4sys.utils.logging_config import RL4SysLogConfig
    debug = args.debug
    config = RL4SysLogConfig.get_default_config(
        log_level="DEBUG" if debug else "INFO",
        structured_logging=True
    )
    RL4SysLogConfig.setup_logging(config_dict=config)

    game = LunarLanderContinuousSim(
        seed=args.seed,
        client_id=args.client_id,
        render_game=args.render,
        debug=debug
    )

    game.logger.info(
        "Starting Lunar Lander Continuous simulation",
        num_iterations=args.number_of_iterations,
        max_moves=args.number_of_moves,
        seed=args.seed,
        render=args.render
    )

    game.run_application(num_iterations=args.number_of_iterations, max_moves=args.number_of_moves)


