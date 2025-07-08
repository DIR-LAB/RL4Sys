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

Structure:
    - Each iteration is an "epoch" containing multiple episodes
    - Each episode processes jobs until environment sends 'done' flag
    - Episodes terminate naturally when all jobs in sequence are scheduled
    - Each epoch collects 100 episodes (traj_per_epoch)
    - This matches the structure of ppo-pick-jobs.py
"""

JOB_FEATURES = 8
MAX_QUEUE_SIZE = 128
JOB_SEQUENCE_SIZE = 256

class JobSchedulingSim:
    """
    Job Scheduling Simulation using HPCSim environment with RL4Sys agent.
    
    This class integrates the HPCSim job scheduling environment with an RL4Sys
    reinforcement learning agent to train and evaluate job scheduling policies.
    
    Attributes:
        _seed (int): Random seed for reproducibility
        _client_id (str): Unique client identifier
        _performance_metric (int): Performance metric type (0-4)
        _workload_file (str): Path to workload file
        logger (StructuredLogger): Logger instance for structured logging
        env (HPCEnv): HPCSim environment instance
        rlagent (RL4SysAgent): RL4Sys reinforcement learning agent
        simulator_stats (dict): Statistics tracking simulation performance
    """
    
    def __init__(self, seed: int, client_id: str, performance_metric: int = 0, workload_file: str = '') -> None:
        """
        Initialize the JobSchedulingSim with environment and agent.
        
        Args:
            seed: Random seed for reproducibility
            client_id: Unique client identifier
            performance_metric: Performance metric type (0-4)
                0: Average bounded slowdown
                1: Average waiting time  
                2: Average turnaround time
                3: Resource utilization
                4: Average slowdown
            workload_file: Path to workload file (SWF format)
        """
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
        self.rlagent = RL4SysAgent(conf_path='./rl4sys/examples/job_schedual_old/job_conf.json')
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

    def build_mask(self, obs: np.ndarray) -> np.ndarray:
        """
        Build action mask based on observation state.
        
        This method implements the same masking logic as in ppo-pick-jobs.py:
        - 0: Invalid action (empty slot or filled slot)
        - 1: Valid action (job available for scheduling)
        
        The observation contains job slots, where each slot has JOB_FEATURES features.
        The masking logic checks for specific patterns:
        - Empty slot pattern: [0, 1, 1, 1, 1, 1, 1, 0] (mask = 0)
        - Filled slot pattern: [1, 1, 1, 1, 1, 1, 1, 1] (mask = 0)  
        - Valid job pattern: Any other pattern (mask = 1)
        
        Args:
            obs: Observation array from environment with shape (MAX_QUEUE_SIZE * JOB_FEATURES,)
            
        Returns:
            Mask array with shape (MAX_QUEUE_SIZE,) where 1 indicates valid actions and 0 indicates invalid actions
        """
        mask = []
        for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
            job_slot = obs[i:i+JOB_FEATURES]
            
            # Check if slot is empty (pattern: [0, 1, 1, 1, 1, 1, 1, 0])
            if all(job_slot == [0] + [1]*(JOB_FEATURES-2) + [0]):
                mask.append(0)
            # Check if slot is filled (pattern: [1, 1, 1, 1, 1, 1, 1, 1])
            elif all(job_slot == [1]*JOB_FEATURES):
                mask.append(0)
            # Valid job available for scheduling
            else:
                mask.append(1)
        
        return np.array(mask, dtype=np.float32)

    def run_application(self, num_iterations: int, max_scheduling_steps: int) -> None:
        """
        Run the job scheduling simulation for specified number of iterations.
        
        This method executes the main simulation loop, collecting experience
        from the environment and training the RL agent through the RL4Sys framework.
        
        Each iteration is structured as an epoch containing multiple episodes,
        where each episode continues until the environment sends a 'done' flag
        (when all jobs in the sequence are scheduled).
        
        Args:
            num_iterations: Number of training iterations (epochs) to run
            max_scheduling_steps: Maximum scheduling steps per episode (unused, kept for compatibility)
        """
        self.logger.info(
            "Starting job scheduling simulation",
            num_iterations=num_iterations,
            max_scheduling_steps=max_scheduling_steps,
            job_sequence_size=JOB_SEQUENCE_SIZE
        )

        profiling = []
        traj_per_epoch = 100  # Number of episodes per epoch (matching ppo-pick-jobs.py default)

        for iteration in range(num_iterations):
            self.simulator_stats['total_iterations'] += 1
            self.logger.info(
                "Starting epoch",
                iteration=iteration,
                total_iterations=self.simulator_stats['total_iterations']
            )

            # Collect multiple episodes for this epoch
            episode_count = 0
            epoch_stats = {
                'episode_returns': [],
                'episode_lengths': [],
                'sjf_scores': [],
                'f1_scores': []
            }

            while episode_count < traj_per_epoch:
                # Reset the environment for new episode
                obs, _ = self.env.reset()
                done = False
                episode_steps = 0
                episode_return = 0
                sjf_score = 0
                f1_score = 0
                start_time = time.time()

                # Build initial observation
                obs_tensor = self.build_observation(obs)

                # Episode loop - let environment determine when episode ends
                while not done:
                    # Build action mask for current observation
                    action_mask = self.build_mask(obs)
                    
                    # Convert mask to torch.Tensor for the agent
                    action_mask_tensor = torch.as_tensor(action_mask, dtype=torch.float32)
                    
                    # Debug logging for mask and action selection
                    self.logger.debug(
                        "Action selection with mask",
                        episode_step=episode_steps,
                        action_mask=action_mask.tolist(),
                        valid_actions_count=np.sum(action_mask),
                        total_actions=len(action_mask)
                    )
                    
                    # Get action from agent with mask
                    self.rl4sys_traj, self.rl4sys_action = self.rlagent.request_for_action(
                        self.rl4sys_traj, 
                        obs_tensor, 
                        mask=action_mask_tensor
                    )
                    self.rlagent.add_to_trajectory(self.rl4sys_traj, self.rl4sys_action)

                    action = self.rl4sys_action.act

                    # Ensure action is compatible and within valid range
                    if isinstance(action, torch.Tensor):
                        action = action.item()
                    elif isinstance(action, np.ndarray):
                        action = action[0]
                    action = int(action)

                    # Validate action is within valid range for HPCSim
                    if action < 0 or action >= MAX_QUEUE_SIZE:
                        self.logger.warning(
                            "Action out of valid range, using default action",
                            action=action,
                            valid_range=(0, MAX_QUEUE_SIZE-1)
                        )
                        action = 0  # Default to no action if out of range

                    # Step the environment - returns [obs, reward, done, reward2, sjf, f1]
                    step_result = self.env.step(action)

                    # Parse step results
                    if len(step_result) == 6:
                        next_obs, reward, done, reward2, sjf_t, f1_t = step_result
                    elif len(step_result) == 4:
                        next_obs, reward, done, reward2 = step_result
                        sjf_t, f1_t = 0, 0

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
                    
                    episode_steps += 1
                    self.simulator_stats['scheduling_decisions'] += 1
                    self.simulator_stats['job_scores'].append(reward)

                    obs_tensor = next_obs_tensor  # Update current observation

                    if done:
                        print(f"step: {episode_steps}, Reward: {reward}, done: {done}, sjf_score: {sjf_score}, f1_score: {f1_score}")

                
                # Episode completed
                self.logger.info(
                    "Episode completed",
                    iteration=iteration,
                    episode=episode_count,
                    episode_steps=episode_steps,
                    episode_return=episode_return,
                    sjf_score=sjf_score,
                    f1_score=f1_score,
                    done=done,
                    expected_steps=JOB_SEQUENCE_SIZE,
                    step_difference=episode_steps - JOB_SEQUENCE_SIZE
                )

                # Mark end of trajectory
                self.rlagent.mark_end_of_trajectory(self.rl4sys_traj, self.rl4sys_action)

                # Record episode statistics
                completion_time = time.time() - start_time
                self.simulator_stats['time_to_completion'].append(completion_time)
                epoch_stats['episode_returns'].append(episode_return)
                epoch_stats['episode_lengths'].append(episode_steps)
                epoch_stats['sjf_scores'].append(sjf_score)
                epoch_stats['f1_scores'].append(f1_score)
                
                # Update global statistics
                self.simulator_stats['episode_returns'].append(episode_return)
                self.simulator_stats['sjf_scores'].append(sjf_score)
                self.simulator_stats['f1_scores'].append(f1_score)
                
                if episode_return > 0:  # Successful scheduling
                    self.simulator_stats['successful_schedules'] += 1
                else:
                    self.simulator_stats['failed_schedules'] += 1

                episode_count += 1

            # Epoch completed - log epoch statistics
            avg_episode_return = np.mean(epoch_stats['episode_returns']) if epoch_stats['episode_returns'] else 0
            avg_episode_length = np.mean(epoch_stats['episode_lengths']) if epoch_stats['episode_lengths'] else 0
            avg_sjf_score = np.mean(epoch_stats['sjf_scores']) if epoch_stats['sjf_scores'] else 0
            avg_f1_score = np.mean(epoch_stats['f1_scores']) if epoch_stats['f1_scores'] else 0

            self.logger.info(
                "Epoch completed",
                iteration=iteration,
                episodes_collected=episode_count,
                avg_episode_return=avg_episode_return,
                avg_episode_length=avg_episode_length,
                avg_sjf_score=avg_sjf_score,
                avg_f1_score=avg_f1_score
            )

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

    def build_observation(self, obs) -> torch.Tensor:
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
                        help='number of epochs to run the job scheduling simulation')
    parser.add_argument('--number-of-steps', type=int, default=256,
                        help='maximum number of scheduling steps per episode (unused, episodes terminate naturally)')
    parser.add_argument('--workload-file', type=str, default='./rl4sys/examples/job_schedual_old/HPCSim/data/lublin_256.swf',
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
        num_epochs=args.number_of_iterations,
        max_steps_per_episode=args.number_of_steps,
        episodes_per_epoch=100,
        seed=args.seed,
        workload_file=args.workload_file,
        performance_metric=args.performance_metric
    )

    job_scheduling_sim.run_application(
        num_iterations=args.number_of_iterations, 
        max_scheduling_steps=args.number_of_steps
    )
