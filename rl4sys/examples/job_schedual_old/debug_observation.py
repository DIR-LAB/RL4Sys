#!/usr/bin/env python3
"""
Debug script to examine observation patterns and mask generation.
"""

import sys
import os
import numpy as np

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

def analyze_observation(obs: np.ndarray, step: int):
    """
    Analyze observation patterns and generate mask.
    """
    print(f"\n=== Step {step} Analysis ===")
    print(f"Observation shape: {obs.shape}")
    print(f"Observation type: {type(obs)}")
    print(f"Observation dtype: {obs.dtype}")
    
    # Show first few values
    print(f"First 20 values: {obs[:20]}")
    
    # Generate mask
    mask = build_mask(obs)
    print(f"Generated mask: {mask}")
    print(f"Valid actions: {np.sum(mask)}/{len(mask)}")
    
    # Analyze first 10 job slots in detail
    print(f"\nFirst 10 job slot patterns:")
    for i in range(min(10, MAX_QUEUE_SIZE)):
        start_idx = i * JOB_FEATURES
        end_idx = start_idx + JOB_FEATURES
        if end_idx <= len(obs):
            job_slot = obs[start_idx:end_idx]
            mask_val = mask[i]
            
            # Check patterns
            is_empty_pattern = all(job_slot == [0] + [1]*(JOB_FEATURES-2) + [0])
            is_filled_pattern = all(job_slot == [1]*JOB_FEATURES)
            
            print(f"  Job {i:2d}: {job_slot} -> Mask={mask_val} (empty={is_empty_pattern}, filled={is_filled_pattern})")
    
    # Check for any non-zero values beyond first slot
    non_zero_indices = np.nonzero(obs)[0]
    print(f"\nNon-zero indices in observation: {non_zero_indices[:50]}{'...' if len(non_zero_indices) > 50 else ''}")
    
    return mask

def main():
    """
    Main debug function.
    """
    print("=== Job Scheduling Observation Debug ===")
    
    # Set seed for reproducibility
    np.random.seed(0)
    
    # Initialize environment
    env = HPCEnv(shuffle=False, backfil=False, skip=False, job_score_type=0, batch_job_slice=0, build_sjf=False)
    env.seed(0)
    
    # Use the same workload file as job_main.py
    workload_file = './rl4sys/examples/job_schedual_old/HPCSim/data/SDSC-SP2-1998-4.2-cln.swf'
    if os.path.exists(workload_file):
        env.my_init(workload_file=workload_file)
    else:
        print(f"Warning: Workload file not found: {workload_file}")
        print("Using default initialization...")
        env.my_init()
    
    print(f"Environment initialized")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset environment
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Analyze initial observation
    mask = analyze_observation(obs, 0)
    
    # Run a few steps to see how observations change
    for step in range(1, 6):
        # Take a random action
        action = 0  # Always choose action 0 for debugging
        next_obs, reward, done, reward2, sjf_t, f1_t = env.step(action)
        
        print(f"\nStep {step}: Action={action}, Reward={reward}, Done={done}")
        
        if next_obs is not None:
            mask = analyze_observation(next_obs, step)
        else:
            print("Episode ended, no next observation")
            break
        
        if done:
            print("Episode completed")
            break
    
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    main() 