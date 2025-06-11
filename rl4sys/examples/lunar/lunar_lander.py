import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import random
import time
import math
import torch
import gymnasium as gym
import gymnasium.spaces
import threading
from rl4sys.client.agent import RL4SysAgent
from rl4sys.utils.util import StructuredLogger




"""
Environment script: Lunar Lander Simulator

Training server parameters:
    kernel_size | MAX_SIZE = 1
    kernel_dim  | FEATURES = 8
    buf_size    | MOVE_SEQUENCE_SIZE * 100 = 50000
"""

INPUT_DIM = 8
ACT_DIM = 4
MOVE_SEQUENCE_SIZE = 500

class LunarLanderSim():
    def __init__(self, seed, client_id, performance_metric=0, render_game=False):
        self._seed = seed
        self._client_id = client_id
        self._performance_metric = performance_metric
        self._render_game = render_game
        self.logger = StructuredLogger("LunarLanderSim", debug=False)

        # Initialize the Gym environment
        self.env = gym.make('LunarLander-v3', continuous=False)

        # Set the seeds for reproducibility
        self.env.reset(seed=self._seed)
        self.env.action_space.seed(self._seed)
        self.env.observation_space.seed(self._seed)
        np.random.seed(self._seed)
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        if isinstance(self.env.action_space, gym.spaces.Box):
            # For continuous action spaces (DDPG, TD3, RPO)
            act_dim = self.env.action_space.shape[0]
            self.act_limit = self.env.action_space.high[0]  # Assuming symmetric limits
        else:
            # For discrete action spaces (DQN)
            act_dim = self.env.action_space.n
            self.act_limit = 1.0

        self.rlagent = RL4SysAgent(conf_path='./rl4sys/examples/lunar/luna_conf.json')
        self.rl4sys_traj = None
        self.rl4sys_action = None

        # To store simulator stats
        self.simulator_stats = {
            'moves': 0,
            'action_rewards': [],
            'performance_rewards': [],
            'success_count': 0,
            'death_count': 0,
            'time_to_goal': [],
            'time_to_death': [],
            'total_iterations': 0
        }

        self.logger.info(
            "Initialized LunarLanderSim",
            client_id=self._client_id,
            seed=self._seed,
            render_game=self._render_game,
            action_space_type="continuous" if isinstance(self.env.action_space, gym.spaces.Box) else "discrete",
            action_dim=act_dim,
            action_limit=self.act_limit
        )

    def run_application(self, num_iterations, max_moves):
        self.logger.info(
            "Starting simulation",
            num_iterations=num_iterations,
            max_moves=max_moves
        )

        profiling = []

        for iteration in range(num_iterations):
            self.simulator_stats['total_iterations'] += 1
            self.logger.info(
                "Starting iteration",
                iteration=iteration,
                total_iterations=self.simulator_stats['total_iterations']
            )

            # Reset the environment with the seed
            obs, _ = self.env.reset(seed=self._seed + iteration)
            done = False
            moves = 0
            cumulative_reward = 0
            start_time = time.time()

            # Build initial observation
            obs_tensor = self.build_observation(obs)

            t0_start_time = time.perf_counter_ns()

            env_ns = 0
            infer_ns = 0

            while not done or moves < max_moves:
                if self._render_game:
                    self.env.render()

                # Get action from agent
                t0 = time.perf_counter_ns()
                self.rl4sys_traj, self.rl4sys_action = self.rlagent.request_for_action(self.rl4sys_traj, obs_tensor)
                infer_ns += time.perf_counter_ns() - t0


                self.rlagent.add_to_trajectory(self.rl4sys_traj, self.rl4sys_action)
                

                action = self.rl4sys_action.act

                # Ensure action is compatible
                if isinstance(action, torch.Tensor):
                    action = action.item()
                elif isinstance(action, np.ndarray):
                    action = action[0]
                action = int(action)

                # Step the environment
                t0 = time.perf_counter_ns()
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                env_ns += time.perf_counter_ns() - t0

                done = terminated or truncated
                cumulative_reward += reward

                # Build next observation
                next_obs_tensor = self.build_observation(next_obs)

                # record reward
                self.rl4sys_action.update_reward(reward)
                
                moves += 1
                self.simulator_stats['moves'] += 1
                self.simulator_stats['action_rewards'].append(reward)

                obs_tensor = next_obs_tensor  # Update current observation
                
                if moves >= max_moves or done:
                    # Flag last action
                    self.logger.info(
                        "Iteration completed",
                        iteration=iteration,
                        moves=moves,
                        cumulative_reward=cumulative_reward,
                        done=done
                    )

                    # If step exceeds MOVE_SEQUENCE_SIZE or done, set done to True
                    self.rlagent.mark_end_of_trajectory(self.rl4sys_traj, self.rl4sys_action)

                    if reward >= 200:  # Successful landing threshold
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

            total_ns = time.perf_counter_ns() - t0_start_time

            total_ms = total_ns/1e6/moves
            env_ms   = env_ns/1e6/moves
            infer_ms = infer_ns/1e6/moves
            over_ms  = total_ms - env_ms - infer_ms
            profiling.append({"steps/s": round(1000/total_ms,1),
                "env_ms": round(env_ms,3),
                "infer_ms": round(infer_ms,3),
                "over_ms": round(over_ms,3)})
            
            

            if self._render_game:
                self.env.close()
        
        # Log final statistics
        self.logger.info(
            "Simulation completed",
            total_iterations=self.simulator_stats['total_iterations'],
            total_moves=self.simulator_stats['moves'],
            success_count=self.simulator_stats['success_count'],
            death_count=self.simulator_stats['death_count'],
            avg_time_to_goal=np.mean(self.simulator_stats['time_to_goal']) if self.simulator_stats['time_to_goal'] else 0,
            avg_time_to_death=np.mean(self.simulator_stats['time_to_death']) if self.simulator_stats['time_to_death'] else 0,
            avg_reward=np.mean(self.simulator_stats['action_rewards']) if self.simulator_stats['action_rewards'] else 0
        )
        
        avg_step_per_second = 0
        avg_env_ms = 0
        avg_infer_ms = 0
        avg_over_ms = 0
        for i in range(len(profiling)):
            print(f"iteration {i}: {profiling[i]}")
            avg_step_per_second += profiling[i]["steps/s"]
            avg_env_ms += profiling[i]["env_ms"]
            avg_infer_ms += profiling[i]["infer_ms"]
            avg_over_ms += profiling[i]["over_ms"]

        print(f"avg_step_per_second: {round(avg_step_per_second/len(profiling), 1)}")
        print(f"avg_env_ms: {round(avg_env_ms/len(profiling), 3)}")
        print(f"avg_infer_ms: {round(avg_infer_ms/len(profiling), 3)}")
        print(f"avg_over_ms: {round(avg_over_ms/len(profiling), 3)}")

        

    def build_observation(self, obs):
        """
        Build the observation for the RL4Sys agent.
        In Lunar Lander, the observation is already provided by the environment.
        """
        # Convert observation to tensor
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return obs_tensor

    def calculate_performance_return(self, elements) -> float:
        pass


if __name__ == '__main__':
    """ Requires gym

    Runs LunarLanderSim instance(s) according to user input parameters.

    Example Usage:
    python lunar_lander.py --tensorboard=True --number-of-iterations=1 --number-of-moves=1000 --render=True
    or
    python lunar_lander.py --tensorboard=True --number-of-iterations=100 --number-of-moves=100000 --render=False

    """
    import argparse

    parser = argparse.ArgumentParser(prog="RL4Sys Lunar Lander Simulator",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random number generation in environment')
    parser.add_argument('--score-type', type=int, default=0,
                        help='0. avg action reward per reward, 1. avg action reward per success, 2. avg action reward per death,\n' +
                             '3. avg action reward per collision, 4. avg action reward per failure, 5. Time-to-Goal, 6. Time-to-Death')
    parser.add_argument('--number-of-iterations', type=int, default=20,
                        help='number of iterations to train the agent')
    parser.add_argument('--number-of-moves', type=int, default=200,
                        help='maximum number of moves allowed per iteration')
    parser.add_argument('--render', type=bool, default=False,
                        help='render the Lunar Lander environment')
    parser.add_argument('--client-id', type=str, default="lunar_lander",
                        help='uniqueclient id for the Lunar Lander simulation')
    
    args, extras = parser.parse_known_args()

    # Create the simulation environment with the agent
    lunar_lander_game = LunarLanderSim(
        seed=args.seed,
        client_id=args.client_id,
        render_game=args.render
    )

    lunar_lander_game.logger.info(
        "Starting Lunar Lander simulation",
        num_iterations=args.number_of_iterations,
        max_moves=args.number_of_moves,
        seed=args.seed,
        render=args.render
    )

    lunar_lander_game.run_application(num_iterations=args.number_of_iterations, max_moves=args.number_of_moves)
