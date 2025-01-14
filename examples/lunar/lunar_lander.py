import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from _common._examples.BaseApplication import ApplicationAbstract

import numpy as np
import random
import time

import math
import torch

import gymnasium as gym
import gymnasium.spaces

from agent import RL4SysAgent
from training_server import TrainingServer

from utils.plot import get_newest_dataset

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


class LunarLanderSim(ApplicationAbstract):
    def __init__(self, seed, model=None, performance_metric=0, render_game=False):
        super().__init__()
        self._seed = seed
        self._performance_metric = performance_metric
        self._render_game = render_game

        # Initialize the Gym environment
        self.env = gym.make('LunarLander-v3')

        # Set the seeds for reproducibility
        self.env.reset(seed=self._seed)
        self.env.action_space.seed(self._seed)
        self.env.observation_space.seed(self._seed)
        np.random.seed(self._seed)
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        self.rlagent = RL4SysAgent(model=model)

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

    def run_application(self, num_iterations=1, num_moves=1000):
        for iteration in range(num_iterations):
            self.simulator_stats['total_iterations'] += 1
            print(f"[LunarLanderSim - simulator] Game Iteration {iteration}")

            # Reset the environment with the seed
            obs, info = self.env.reset(seed=self._seed + iteration)
            done = False
            moves = 0
            rl_runs = 0
            cumulative_reward = 0
            start_time = time.time()

            # Build initial observation
            obs_tensor, mask = self.build_observation(obs)

            # while not done and moves < num_moves: # Modified
            while not done:  
                if self._render_game:
                    self.env.render()

                # Get action from agent
                rl4sys_action = self.rlagent.request_for_action(obs_tensor, mask)
                action = rl4sys_action.act

                # Ensure action is compatible
                if isinstance(action, torch.Tensor):
                    action = action.item()
                elif isinstance(action, np.ndarray):
                    action = action[0]
                action = int(action)

                # Step the environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                cumulative_reward += reward

                # Build next observation
                next_obs_tensor, mask = self.build_observation(next_obs)

                # record trajectory
                rl4sys_action.update_reward(reward)
                rl_runs += 1
                moves += 1
                self.simulator_stats['moves'] += 1
                self.simulator_stats['action_rewards'].append(reward)

                obs_tensor = next_obs_tensor  # Update current observation

                
                if rl_runs >= MOVE_SEQUENCE_SIZE:
                    # Flag last action
                    print(f'[LunarLanderSim - simulator] RL4SysAgent moves made: {moves}')
                    self.simulator_stats['moves'] = moves
                    rl_runs = 0
                    rl_total = self.calculate_performance_return(self.simulator_stats)
                    rew = -rl_total
                    self.simulator_stats['performance_rewards'].append(rew)
                    self.rlagent.flag_last_action(rew)
                    break
                

                if done:
                    print(f'[LunarLanderSim - simulator] RL4SysAgent moves made: {moves}')
                    self.simulator_stats['moves'] = moves
                    rl_runs = 0
                    rl_total = self.calculate_performance_return(self.simulator_stats)
                    rew = -rl_total
                    self.rlagent.flag_last_action(rew)
                    if reward >= 200:  # Successful landing threshold
                        self.simulator_stats['success_count'] += 1
                        self.simulator_stats['time_to_goal'].append(time.time() - start_time)
                    else:
                        self.simulator_stats['death_count'] += 1
                        self.simulator_stats['time_to_death'].append(time.time() - start_time)
                    break

            if self._render_game:
                self.env.close()
        
        

    def build_observation(self, obs):
        """
        Build the observation for the RL4Sys agent.
        In Lunar Lander, the observation is already provided by the environment.
        We'll also create a mask for action space.
        """
        # Convert observation to tensor
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        # Create mask for action space (discrete actions from 0 to 3)
        mask = torch.zeros(ACT_DIM, dtype=torch.float32).unsqueeze(0)

        return obs_tensor, mask

    def calculate_performance_return(self, elements) -> float:
        """
        Calculate performance score based on performance metric using captured simulator elements
        :return: returns calculated performance score
        """
        if self._performance_metric == 0:
            # avg cumulative reward per reward count
            return float(sum(elements['action_rewards']) / len(elements['action_rewards']))
        elif self._performance_metric == 1:
            # avg cumulative reward per success rate
            if elements['success_count'] > 0:
                return float(sum(elements['action_rewards']) / elements['success_count'])
            else:
                return 0.0
        elif self._performance_metric == 2:
            # avg cumulative reward per death rate
            if elements['death_count'] > 0:
                return -float(sum(elements['action_rewards']) / elements['death_count'])
            else:
                return 0.0
        elif self._performance_metric == 3:
            # avg cumulative reward per collision rate (using death_count in Lunar Lander)
            if elements['death_count'] > 0:
                return -float(sum(elements['action_rewards']) / elements['death_count'])
            else:
                return 0.0
        elif self._performance_metric == 4:
            # avg cumulative reward per failure rate
            total_failures = elements['death_count']
            if total_failures > 0:
                return float(sum(elements['action_rewards']) / total_failures)
            else:
                return 0.0
        elif self._performance_metric == 5:
            # avg cumulative time-to-goal per success
            if elements['success_count'] > 0:
                return float((sum(elements['time_to_goal']) / elements['success_count']))
            else:
                return 0.0
        elif self._performance_metric == 6:
            # time-to-death per death
            if elements['death_count'] > 0:
                return -float(sum(elements['time_to_death']) / elements['death_count'])
            else:
                return 0.0
        else:
            raise NotImplementedError


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
    parser.add_argument('--model-path', type=str, default=None,
                        help='path to pre-existing model to be loaded by agent')
    parser.add_argument('--tensorboard', type=bool, default=True,
                        help='enable tensorboard logging for training observations and insights.\n' +
                             'Make sure to properly configure tensorboard parameters in config.json before running.')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for random number generation in environment')
    parser.add_argument('--score-type', type=int, default=0,
                        help='0. avg action reward per reward, 1. avg action reward per success, 2. avg action reward per death,\n' +
                             '3. avg action reward per collision, 4. avg action reward per failure, 5. Time-to-Goal, 6. Time-to-Death')
    parser.add_argument('--number-of-iterations', type=int, default=10000,
                        help='number of iterations to train the agent')
    parser.add_argument('--number-of-moves', type=int, default=10000,
                        help='maximum number of moves allowed per iteration')
    parser.add_argument('--start-server', '-s', dest='algorithm', type=str, default='DQN',
                        help='run a local training server, using a specific algorithm')
    parser.add_argument('--render', type=bool, default=False,
                        help='render the Lunar Lander environment')
    args, extras = parser.parse_known_args()

    # start training server
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if args.algorithm != 'No Server':
        extras.append('--buf_size')
        extras.append(str(MOVE_SEQUENCE_SIZE * 100))
        rl_training_server = TrainingServer(algorithm_name=args.algorithm, 
                                            input_size=INPUT_DIM, 
                                            action_dim=ACT_DIM, 
                                            hyperparams=extras, 
                                            env_dir=app_dir, 
                                            tensorboard=args.tensorboard)
        print('[lunar_lander.py] Created Training Server')

    # load model if applicable
    model_arg = torch.load(args.model_path, map_location=torch.device('cpu')) if args.model_path else None

    # create simulation environment
    lunar_lander_game = LunarLanderSim(args.seed, model=model_arg,
                                       performance_metric=args.score_type,
                                       render_game=args.render)

    # run simulation
    print(f"[lunar_lander.py] Running {args.number_of_iterations} iterations...")
    lunar_lander_game.run_application(num_iterations=args.number_of_iterations, num_moves=args.number_of_moves)
