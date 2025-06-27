import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import random
import time
import math
import torch
import threading
from rl4sys.client.agent import RL4SysAgent
from rl4sys.utils.util import StructuredLogger
from rl4sys.utils.logging_config import setup_rl4sys_logging

# Set up logging with debug enabled if requested
# setup_rl4sys_logging(debug=True)

# Import HPCSim components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'HPCSim')))
from HPCSim.job import Job, Workloads
from HPCSim.cluster import Cluster
from HPCSim.HPCSimPickJobs import HPCEnv

"""
Job Scheduling Environment Script using HPCSim

Training server parameters:
    kernel_size | MAX_SIZE = 1
    kernel_dim  | JOB_FEATURES = 8
    buf_size    | JOB_SEQUENCE_SIZE * 100 = 25600
"""

JOB_FEATURES = 8
MAX_QUEUE_SIZE = 128
JOB_SEQUENCE_SIZE = 256

class JobSchedulingSim():
    def __init__(self, seed, client_id, performance_metric=0, workload_file=''):
        self._seed = seed
        self._client_id = client_id
        self._performance_metric = performance_metric
        self._workload_file = workload_file
        self.logger = StructuredLogger("JobSchedulingSim", debug=True) # TODO debug only

        # Set the seeds for reproducibility
        np.random.seed(self._seed)
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        # Initialize the HPCSim environment
        self.env = HPCEnv(
            shuffle=False, 
            backfil=False, 
            skip=False, 
            job_score_type=self._performance_metric,
            batch_job_slice=0,
            build_sjf=False
        )

        
        # Load workload if provided
        if self._workload_file:
            self.env.my_init(workload_file=self._workload_file)
        else:
            # Use default workload path
            default_workload = os.path.join(os.path.dirname(__file__), 'HPCSim', 'lublin_256.swf')
            if os.path.exists(default_workload):
                self.env.my_init(workload_file=default_workload)

        # Initialize RL4Sys agent
        self.rlagent = RL4SysAgent(conf_path='/Users/girigiri_yomi/Udel_Proj/RL4Sys/rl4sys/examples/job_schedual_old/job_conf.json')
        self.rl4sys_traj = None
        self.rl4sys_action = None

        # To store simulator stats
        self.simulator_stats = {
            'scheduling_decisions': 0,
            'job_scores': [],
            'performance_rewards': [],
            'successful_schedules': 0,
            'failed_schedules': 0,
            'time_to_completion': [],
            'resource_utilization': [],
            'total_iterations': 0,
            'episode_returns': [],
            'sjf_scores': [],
            'f1_scores': []
        }

        self.logger.info(
            "Initialized JobSchedulingSim",
            client_id=self._client_id,
            seed=self._seed,
            workload_file=self._workload_file,
            job_features=JOB_FEATURES,
            max_queue_size=MAX_QUEUE_SIZE,
            job_sequence_size=JOB_SEQUENCE_SIZE
        )

    def run_application(self, num_iterations, max_scheduling_steps):
        self.logger.info(
            "Starting job scheduling simulation",
            num_iterations=num_iterations,
            max_scheduling_steps=max_scheduling_steps
        )

        profiling = []

        for iteration in range(num_iterations):
            self.simulator_stats['total_iterations'] += 1
            self.logger.info(
                "Starting iteration",
                iteration=iteration,
                total_iterations=self.simulator_stats['total_iterations']
            )

            # Reset the environment
            obs,_ = self.env.reset()
            done = False
            scheduling_steps = 0
            cumulative_reward = 0
            episode_return = 0
            sjf_score = 0
            f1_score = 0
            start_time = time.time()

            # Build initial observation
            obs_tensor = self.build_observation(obs)

            t0_start_time = time.perf_counter_ns()

            env_ns = 0
            infer_ns = 0
            step = 0

            while True:
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

                # Ensure action is within valid range for HPCSim
                if action >= MAX_QUEUE_SIZE:
                    action = 0  # Default to no action if out of range

                # Step the environment - returns [obs, reward, done, reward2, sjf, f1]
                t0 = time.perf_counter_ns()
                step_result = self.env.step(action)
                env_ns += time.perf_counter_ns() - t0

                # Parse step results
                if len(step_result) == 6:
                    next_obs, reward, done, reward2, sjf_t, f1_t = step_result
                elif len(step_result) == 4:
                    next_obs, reward, done, reward2 = step_result
                    sjf_t, f1_t = 0, 0

                cumulative_reward += reward
                episode_return += reward
                sjf_score += sjf_t
                f1_score += f1_t

                # Build next observation
                if next_obs is not None:
                    next_obs_tensor = self.build_observation(next_obs)
                else:
                    # Episode ended, no next observation
                    next_obs_tensor = None

                # Record reward
                self.rl4sys_action.update_reward(reward)
                
                scheduling_steps += 1
                self.simulator_stats['scheduling_decisions'] += 1
                self.simulator_stats['job_scores'].append(reward)

                obs_tensor = next_obs_tensor  # Update current observation

                step += 1
                
                #if scheduling_steps >= max_scheduling_steps or done:
                if done:
                    print(f"step: {step}, done: {done}, reward: {reward}, reward2: {reward2}, sjf_t: {sjf_t}, f1_t: {f1_t}")
                    # Flag last action
                    self.logger.info(
                        "Iteration completed",
                        iteration=iteration,
                        scheduling_steps=scheduling_steps,
                        cumulative_reward=cumulative_reward,
                        episode_return=episode_return,
                        sjf_score=sjf_score,
                        f1_score=f1_score,
                        done=done
                    )

                    # Mark end of trajectory
                    self.rlagent.mark_end_of_trajectory(self.rl4sys_traj, self.rl4sys_action)

                    # Record performance metrics
                    completion_time = time.time() - start_time
                    self.simulator_stats['time_to_completion'].append(completion_time)
                    self.simulator_stats['episode_returns'].append(episode_return)
                    self.simulator_stats['sjf_scores'].append(sjf_score)
                    self.simulator_stats['f1_scores'].append(f1_score)
                    
                    if episode_return > 0:  # Successful scheduling
                        self.simulator_stats['successful_schedules'] += 1
                        self.logger.info(
                            "Successful scheduling iteration",
                            iteration=iteration,
                            completion_time=completion_time,
                            successful_schedules=self.simulator_stats['successful_schedules']
                        )
                    else:
                        self.simulator_stats['failed_schedules'] += 1
                        self.logger.info(
                            "Failed scheduling iteration",
                            iteration=iteration,
                            completion_time=completion_time,
                            failed_schedules=self.simulator_stats['failed_schedules']
                        )
                    break

            total_ns = time.perf_counter_ns() - t0_start_time

            total_ms = total_ns/1e6/scheduling_steps if scheduling_steps > 0 else 0
            env_ms   = env_ns/1e6/scheduling_steps if scheduling_steps > 0 else 0
            infer_ms = infer_ns/1e6/scheduling_steps if scheduling_steps > 0 else 0
            over_ms  = total_ms - env_ms - infer_ms
            profiling.append({
                "steps/s": round(1000/total_ms,1) if total_ms > 0 else 0,
                "env_ms": round(env_ms,3),
                "infer_ms": round(infer_ms,3),
                "over_ms": round(over_ms,3)
            })

        # Log final statistics
        self.logger.info(
            "Simulation completed",
            total_iterations=self.simulator_stats['total_iterations'],
            total_scheduling_decisions=self.simulator_stats['scheduling_decisions'],
            successful_schedules=self.simulator_stats['successful_schedules'],
            failed_schedules=self.simulator_stats['failed_schedules'],
            avg_time_to_completion=np.mean(self.simulator_stats['time_to_completion']) if self.simulator_stats['time_to_completion'] else 0,
            avg_job_score=np.mean(self.simulator_stats['job_scores']) if self.simulator_stats['job_scores'] else 0,
            avg_episode_return=np.mean(self.simulator_stats['episode_returns']) if self.simulator_stats['episode_returns'] else 0,
            avg_sjf_score=np.mean(self.simulator_stats['sjf_scores']) if self.simulator_stats['sjf_scores'] else 0,
            avg_f1_score=np.mean(self.simulator_stats['f1_scores']) if self.simulator_stats['f1_scores'] else 0
        )
        
        """
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

        if len(profiling) > 0:
            print(f"avg_step_per_second: {round(avg_step_per_second/len(profiling), 1)}")
            print(f"avg_env_ms: {round(avg_env_ms/len(profiling), 3)}")
            print(f"avg_infer_ms: {round(avg_infer_ms/len(profiling), 3)}")
            print(f"avg_over_ms: {round(avg_over_ms/len(profiling), 3)}")
        """
        

    def build_observation(self, obs):
        """
        Build the observation for the RL4Sys agent.
        Convert the HPCSim observation to tensor format.
        """
        # Convert observation to tensor
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        else:
            # Handle other observation formats
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return obs_tensor

    def calculate_performance_return(self, elements) -> float:
        """
        Calculate performance return based on job scheduling metrics.
        This can be customized based on specific performance objectives.
        """
        pass


if __name__ == '__main__':
    """ 
    Runs JobSchedulingSim instance(s) according to user input parameters.

    Example Usage:
    python job_main.py --seed=1 --number-of-iterations=20 --number-of-steps=100 --workload-file=HPCSim/lublin_256.swf
    or
    python job_main.py --seed=1 --number-of-iterations=100 --number-of-steps=1000 --workload-file=HPCSim/lublin_256.swf
    """
    import argparse

    parser = argparse.ArgumentParser(prog="RL4Sys Job Scheduling Simulator",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random number generation in environment')
    parser.add_argument('--performance-metric', type=int, default=0,
                        help='0: Average bounded slowdown, 1: Average waiting time,\n' +
                             '2: Average turnaround time, 3: Resource utilization, 4: Average slowdown')
    parser.add_argument('--number-of-iterations', type=int, default=10000,
                        help='number of iterations to run the job scheduling simulation')
    parser.add_argument('--number-of-steps', type=int, default=100,
                        help='maximum number of scheduling steps allowed per iteration')
    parser.add_argument('--workload-file', type=str, default='/Users/girigiri_yomi/Udel_Proj/RL4Sys/rl4sys/examples/job_schedual_old/HPCSim/data/lublin_256.swf',
                        help='path to the workload file (SWF format)')
    parser.add_argument('--client-id', type=str, default="job_schedualing_old",
                        help='unique client id for the job scheduling simulation')
    
    args, extras = parser.parse_known_args()

    # Create the simulation environment with the agent
    job_scheduling_sim = JobSchedulingSim(
        seed=args.seed,
        client_id=args.client_id,
        performance_metric=args.performance_metric,
        workload_file=args.workload_file
    )

    job_scheduling_sim.logger.info(
        "Starting Job Scheduling simulation",
        num_iterations=args.number_of_iterations,
        max_steps=args.number_of_steps,
        seed=args.seed,
        workload_file=args.workload_file,
        performance_metric=args.performance_metric
    )

    job_scheduling_sim.run_application(
        num_iterations=args.number_of_iterations, 
        max_scheduling_steps=args.number_of_steps
    )
