#!/usr/bin/env python3
"""
Test script to force different actions and see if more jobs appear in the queue.
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

def test_force_different_actions():
    """
    Test forcing different actions to see if more jobs appear in the queue.
    """
    print("=== Testing Force Different Actions ===")
    
    # Set seed for reproducibility
    np.random.seed(0)
    random.seed(0)
    
    # Initialize environment
    env = HPCEnv(shuffle=False, backfil=False, skip=False, job_score_type=0, batch_job_slice=0, build_sjf=False)
    env.seed(0)
    
    workload_file = './rl4sys/examples/job_schedual_old/HPCSim/data/SDSC-SP2-1998-4.2-cln.swf'
    env.my_init(workload_file=workload_file, sched_file='')
    
    print(f"Environment initialized")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset environment
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    step_count = 0
    max_steps = 50
    
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
        
        # Force different actions to see if environment progresses
        if step_count < 10:
            # First 10 steps: always choose action 0 (like the agent)
            action = 0
            print(f"Choosing action 0 (like agent)")
        elif step_count < 20:
            # Next 10 steps: choose random valid action
            if len(valid_indices) > 0:
                action = np.random.choice(valid_indices)
            else:
                action = 0
            print(f"Choosing random valid action: {action}")
        else:
            # After 20 steps: try to force action 1 if valid
            if mask[1] > 0.5:
                action = 1
                print(f"Choosing action 1")
            elif len(valid_indices) > 0:
                action = np.random.choice(valid_indices)
            else:
                action = 0
            print(f"Choosing action: {action}")
        
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

if __name__ == "__main__":
    test_force_different_actions() 