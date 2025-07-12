import numpy as np
import sys
import os
import time

# Add the HPCSim directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'HPCSim'))

from HPCSim.HPCSimPickJobs import HPCEnv

# Constants from ppo-pick-jobs
MAX_QUEUE_SIZE = 128  # From HPCSimPickJobs.py
JOB_FEATURES = 8      # From HPCSimPickJobs.py
JOB_SEQUENCE_SIZE = 256  # From HPCSimPickJobs.py

def create_mask(obs: np.ndarray) -> np.ndarray:
    """
    Create mask using the exact same logic as ppo-pick-jobs.py.
    
    Args:
        obs: Observation array from environment
        
    Returns:
        mask: Binary mask array indicating valid actions
    """
    lst = []
    for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
        if all(obs[i:i+JOB_FEATURES] == [0]+[1]*(JOB_FEATURES-2)+[0]):
            lst.append(0)  # Invalid job pattern
        elif all(obs[i:i+JOB_FEATURES] == [1]*JOB_FEATURES):
            lst.append(0)  # Invalid job pattern (all ones)
        else:
            lst.append(1)  # Valid job
    return np.array(lst)

def dummy_trainer():
    """
    Dummy trainer that always makes action 0, using the same mask scheme as ppo-pick-jobs.
    """
    # Set seed for reproducibility
    np.random.seed(0)
    
    # Initialize environment with same parameters as ppo-pick-jobs
    env = HPCEnv(shuffle=0, backfil=0, skip=0, job_score_type=0, batch_job_slice=0, build_sjf=False)
    env.seed(0)
    
    # Use the same workload and schedule files as in the original
    workload_file = './rl4sys/examples/job_schedual_old/HPCSim/data/SDSC-SP2-1998-4.2-cln.swf'
    model_path = './rl4sys/examples/job_schedual_old/HPCSim/data/SDSC-SP2-1998-4.2-cln.schd'
    env.my_init(workload_file=workload_file, sched_file=model_path)
    
    print("Starting dummy trainer with always action 0...")
    print("=" * 60)
    
    # Training parameters (similar to ppo-pick-jobs)
    traj_per_epoch = 10  # Reduced for testing - just one trajectory
    epochs = 1  # Just one epoch for detailed analysis
    max_ep_len = 1000
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        
        # Episode tracking
        t = 0  # Number of trajectories in this epoch
        epoch_stats = {
            'ep_ret': [],
            'ep_len': [],
            'show_ret': [],
            'sjf': [],
            'f1': []
        }
        
        while t < traj_per_epoch:
            # Reset environment
            obs, cobs = env.reset()
            
            # Episode variables
            ep_ret = 0
            ep_len = 0
            show_ret = 0
            sjf = 0
            f1 = 0
            
            print(f"  Starting trajectory {t + 1}/{traj_per_epoch}")
            print(f"  Initial observation shape: {obs.shape}")
            
            # Episode loop
            while ep_len < max_ep_len:
                print(f"\n    --- Step {ep_len} ---")
                
                # Create mask using the same logic as ppo-pick-jobs
                mask = create_mask(obs)
                
                # Detailed mask analysis
                valid_actions = np.sum(mask)
                valid_indices = np.where(mask == 1)[0]
                invalid_indices = np.where(mask == 0)[0]
                
                print(f"    Mask shape: {mask.shape}")
                print(f"    Valid actions: {valid_actions}/{len(mask)}")
                print(f"    Valid action indices: {valid_indices[:20]}{'...' if len(valid_indices) > 20 else ''}")
                print(f"    Invalid action indices: {invalid_indices[:20]}{'...' if len(invalid_indices) > 20 else ''}")
                
                # Show first few mask values for debugging
                print(f"    First 10 mask values: {mask[:10]}")
                
                # Show observation patterns for first few jobs
                print(f"    First 3 job feature patterns:")
                for i in range(min(3, MAX_QUEUE_SIZE)):
                    start_idx = i * JOB_FEATURES
                    end_idx = start_idx + JOB_FEATURES
                    if end_idx <= len(obs):
                        job_features = obs[start_idx:end_idx]
                        print(f"      Job {i}: {job_features} -> Mask[{i}] = {mask[i]}")
                
                # Always choose action 0 (as requested)
                action = 0
                
                # Check if action 0 is valid
                if action < len(mask):
                    is_valid = mask[action] == 1
                    print(f"    Action {action} is {'VALID' if is_valid else 'INVALID'} - mask[{action}] = {mask[action]}")
                    
                    if not is_valid:
                        print(f"    WARNING: Action {action} is invalid! Available valid actions: {valid_indices[:10]}")
                else:
                    print(f"    ERROR: Action {action} is out of bounds! Mask length: {len(mask)}")
                
                # Take step in environment
                step_result = env.step(action)
                obs, r, d, r2, sjf_t, f1_t = step_result
                
                print(f"    Step result: reward={r}, done={d}")
                
                # Update episode stats
                ep_ret += r
                ep_len += 1
                show_ret += r2
                sjf += sjf_t
                f1 += f1_t
                
                # Check if episode is done
                if d:
                    print(f"    Episode finished after {ep_len} steps")
                    print(f"    Final reward: {ep_ret}")
                    break
                
                # Limit output for long episodes
                if ep_len >= 10:
                    print(f"    ... continuing for {max_ep_len - ep_len} more steps ...")
                    break
            
            # Store episode statistics
            epoch_stats['ep_ret'].append(ep_ret)
            epoch_stats['ep_len'].append(ep_len)
            epoch_stats['show_ret'].append(show_ret)
            epoch_stats['sjf'].append(sjf)
            epoch_stats['f1'].append(f1)
            
            t += 1
        
        # Log epoch statistics
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Episode Return: {np.mean(epoch_stats['ep_ret']):.4f}")
        print(f"  Average Episode Length: {np.mean(epoch_stats['ep_len']):.2f}")
        print(f"  Average Show Return: {np.mean(epoch_stats['show_ret']):.4f}")
        print(f"  Average SJF: {np.mean(epoch_stats['sjf']):.4f}")
        print(f"  Average F1: {np.mean(epoch_stats['f1']):.4f}")
        print(f"  Total trajectories: {len(epoch_stats['ep_ret'])}")
    
    total_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"Dummy trainer completed in {total_time:.2f} seconds")
    print(f"Total epochs: {epochs}")
    print(f"Trajectories per epoch: {traj_per_epoch}")
    print(f"Always used action: 0")

def simple_test():
    """
    Simple test to verify observation and action space sizes.
    """
    # Set seed for reproducibility
    np.random.seed(0)
    
    # Initialize environment
    env = HPCEnv(shuffle=0, backfil=0, skip=0, job_score_type=0, batch_job_slice=0, build_sjf=False)
    env.seed(0)
    
    # Use the same workload and schedule files
    workload_file = './rl4sys/examples/job_schedual_old/HPCSim/data/SDSC-SP2-1998-4.2-cln.swf'
    model_path = './rl4sys/examples/job_schedual_old/HPCSim/data/SDSC-SP2-1998-4.2-cln.schd'
    env.my_init(workload_file=workload_file, sched_file=model_path)
    
    print("=== Simple Test ===")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"MAX_QUEUE_SIZE: {MAX_QUEUE_SIZE}")
    print(f"JOB_FEATURES: {JOB_FEATURES}")
    print(f"Expected observation size: {MAX_QUEUE_SIZE * JOB_FEATURES}")
    
    # Reset environment
    obs, cobs = env.reset()
    print(f"Actual observation size: {len(obs)}")
    
    # Create mask
    mask = create_mask(obs)
    print(f"Mask size: {len(mask)}")
    print(f"Valid actions: {np.sum(mask)}/{len(mask)}")
    
    # Show first few job patterns
    print("\nFirst 5 job patterns:")
    for i in range(5):
        start_idx = i * JOB_FEATURES
        end_idx = start_idx + JOB_FEATURES
        job_features = obs[start_idx:end_idx]
        print(f"  Job {i}: {job_features} -> Valid: {mask[i]}")

def test_valid_actions_over_time():
    """
    Test how the number of valid actions changes over time.
    """
    # Set seed for reproducibility
    np.random.seed(0)
    
    # Initialize environment
    env = HPCEnv(shuffle=0, backfil=0, skip=0, job_score_type=0, batch_job_slice=0, build_sjf=False)
    env.seed(0)
    
    # Use the same workload and schedule files
    workload_file = './rl4sys/examples/job_schedual_old/HPCSim/data/SDSC-SP2-1998-4.2-cln.swf'
    model_path = './rl4sys/examples/job_schedual_old/HPCSim/data/SDSC-SP2-1998-4.2-cln.schd'
    env.my_init(workload_file=workload_file, sched_file=model_path)
    
    print("=== Testing Valid Actions Over Time ===")
    
    # Reset environment
    obs, cobs = env.reset()
    
    step_count = 0
    max_steps = 50
    
    while step_count < max_steps:
        # Create mask
        mask = create_mask(obs)
        valid_actions = np.sum(mask)
        valid_indices = np.where(mask == 1)[0]
        
        print(f"Step {step_count}: Valid actions: {valid_actions}/{len(mask)}")
        if valid_actions > 0:
            print(f"  Valid action indices: {valid_indices[:10]}{'...' if len(valid_indices) > 10 else ''}")
        
        # Always choose action 0
        action = 0
        
        # Check if action 0 is valid
        if action < len(mask):
            is_valid = mask[action] == 1
            print(f"  Action {action} is {'VALID' if is_valid else 'INVALID'}")
            
            if not is_valid and valid_actions > 0:
                print(f"  Available valid actions: {valid_indices[:5]}")
        else:
            print(f"  Action {action} is out of bounds!")
        
        # Take step
        step_result = env.step(action)
        obs, r, d, r2, sjf_t, f1_t = step_result
        
        print(f"  Reward: {r}, Done: {d}")
        
        if d:
            print(f"Episode finished after {step_count + 1} steps")
            break
        
        step_count += 1
        
        # Limit output
        if step_count >= 20:
            print(f"... continuing for {max_steps - step_count} more steps ...")
            break


if __name__ == "__main__":
    # Run simple test first
    simple_test()
    
    print("\n" + "=" * 60)
    
    # Run valid actions over time test
    test_valid_actions_over_time()
    
    print("\n" + "=" * 60)
    
    # Run the dummy trainer
    dummy_trainer()
    
    # Optionally run the original mask test as well
    # test_mask()
