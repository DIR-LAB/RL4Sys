#!/usr/bin/env python3
"""
Test script to verify proper job scheduling and mask generation.
"""

import sys
import os
import numpy as np
import random

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Add HPCSim to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'HPCSim')))

from HPCSim.HPCSimPickJobs import HPCEnv

# Constants
JOB_FEATURES = 8
MAX_QUEUE_SIZE = 128
JOB_SEQUENCE_SIZE = 256

def build_mask(obs: np.ndarray) -> np.ndarray:
    """
    Build action mask based on observation state (same as job_main.py).
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

def test_proper_scheduling():
    """
    Test that the environment works correctly when jobs are properly scheduled.
    """
    print("=== Testing Proper Job Scheduling ===")
    
    # Set seed for reproducibility
    np.random.seed(0)
    random.seed(0)
    
    # Initialize environment with same parameters as ppo-pick-jobs
    env = HPCEnv(shuffle=False, backfil=False, skip=False, job_score_type=0, batch_job_slice=0, build_sjf=False)
    env.seed(0)
    
    # Use the same workload file
    workload_file = './rl4sys/examples/job_schedual_old/HPCSim/data/SDSC-SP2-1998-4.2-cln.swf'
    env.my_init(workload_file=workload_file, sched_file='')
    
    print(f"Environment initialized")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset environment
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Generate initial mask
    mask = build_mask(obs)
    print(f"Initial mask: {mask}")
    print(f"Valid actions: {np.sum(mask)}/{len(mask)}")
    
    step_count = 0
    max_steps = 20
    
    while step_count < max_steps:
        print(f"\n--- Step {step_count} ---")
        
        # Generate mask
        mask = build_mask(obs)
        valid_actions = np.sum(mask)
        valid_indices = np.where(mask == 1)[0]
        
        print(f"Valid actions: {valid_actions}/{len(mask)}")
        print(f"Valid action indices: {valid_indices}")
        
        if valid_actions == 0:
            print("No valid actions available!")
            break
        
        # Choose a valid action (not always 0)
        if len(valid_indices) > 0:
            # Choose a random valid action instead of always 0
            action = np.random.choice(valid_indices)
        else:
            action = 0
        
        print(f"Chosen action: {action}")
        
        # Step the environment
        next_obs, reward, done, reward2, sjf_t, f1_t = env.step(action)
        
        print(f"Reward: {reward}, Done: {done}, SJF: {sjf_t}, F1: {f1_t}")
        
        if next_obs is not None:
            obs = next_obs
        else:
            print("Episode ended, no next observation")
            break
        
        if done:
            print("Episode completed")
            break
        
        step_count += 1
    
    print("\n=== Test Complete ===")

def test_multiple_episodes():
    """
    Test multiple episodes to see if the environment behaves consistently.
    """
    print("\n=== Testing Multiple Episodes ===")
    
    # Set seed for reproducibility
    np.random.seed(0)
    random.seed(0)
    
    # Initialize environment
    env = HPCEnv(shuffle=False, backfil=False, skip=False, job_score_type=0, batch_job_slice=0, build_sjf=False)
    env.seed(0)
    
    workload_file = './rl4sys/examples/job_schedual_old/HPCSim/data/SDSC-SP2-1998-4.2-cln.swf'
    env.my_init(workload_file=workload_file, sched_file='')
    
    for episode in range(1):
        print(f"\n--- Episode {episode} ---")
        
        # Reset environment
        obs, _ = env.reset()
        
        # Generate mask
        mask = build_mask(obs)
        valid_actions = np.sum(mask)
        
        print(f"Initial valid actions: {valid_actions}/{len(mask)}")
        
        step_count = 0
        max_steps = 256
        
        done = False
        while not done:
            mask = build_mask(obs)
            valid_indices = np.where(mask == 1)[0]
            
            """
            if len(valid_indices) > 0:
                action = np.random.choice(valid_indices)
            else:
                action = 0
            """
            action = 0
            
            next_obs, reward, done, reward2, sjf_t, f1_t = env.step(action)
            
            print(f"  Step {step_count}: Action={action}, Valid={len(valid_indices)}, Reward={reward:.4f}, Done={done}")
            
            if next_obs is not None:
                obs = next_obs
            else:
                break
            
            if done:
                break
            
            step_count += 1

if __name__ == "__main__":
    #test_proper_scheduling()
    test_multiple_episodes() 