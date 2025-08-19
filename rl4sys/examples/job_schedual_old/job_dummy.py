import sys
import os
import numpy as np
import random

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import HPCSim components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'HPCSim')))
from HPCSim.HPCSimPickJobs import HPCEnv

# Constants
JOB_SEQUENCE_SIZE = 256
MAX_QUEUE_SIZE = 128

def test_hpcenv_episode_length_exact_config(num_episodes=5, seed=42):
    """
    Test HPCEnv with the exact config as ppo-pick-jobs.py.
    """
    print(f"Testing HPCEnv episode lengths with {num_episodes} episodes...")
    print(f"Expected episode length: {JOB_SEQUENCE_SIZE} steps (if agent is optimal)")
    print("-" * 60)
    
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize environment with exact config
    env = HPCEnv(
        shuffle=False,
        backfil=False,
        skip=False,
        job_score_type=0,
        batch_job_slice=0,
        build_sjf=False
    )
    
    # Load workload
    workload_file = os.path.join(os.path.dirname(__file__), 'HPCSim', 'data', 'lublin_256.swf')
    if not os.path.exists(workload_file):
        print(f"Workload file not found: {workload_file}")
        return
    env.my_init(workload_file=workload_file)
    
    episode_lengths = []
    episode_returns = []
    
    # Run episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        print(f"Episode {episode + 1}: ", end="")
        
        # Run episode
        while not done:
            # Take random action
            action = random.randint(0, MAX_QUEUE_SIZE - 1)
            
            # Step environment
            step_result = env.step(action)
            
            if len(step_result) == 6:
                next_obs, reward, done, reward2, sjf_t, f1_t = step_result
            elif len(step_result) == 4:
                next_obs, reward, done, reward2 = step_result
            
            steps += 1
            total_reward += reward
            
            # Safety check to prevent infinite loops
            if steps > 2000:
                print(f"WARNING: Episode exceeded 2000 steps, forcing termination")
                done = True
                break
        
        episode_lengths.append(steps)
        episode_returns.append(total_reward)
        
        # Print episode results
        expected_diff = steps - JOB_SEQUENCE_SIZE
        status = "✓" if steps == JOB_SEQUENCE_SIZE else "✗"
        print(f"Steps: {steps:4d}, Reward: {total_reward:8.2f}, Diff: {expected_diff:+4d} {status}")
    
    # Print summary statistics
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    min_length = np.min(episode_lengths)
    max_length = np.max(episode_lengths)
    exact_256_count = sum(1 for length in episode_lengths if length == JOB_SEQUENCE_SIZE)
    
    print(f"\nSummary:")
    print(f"  Average episode length: {avg_length:.1f} ± {std_length:.1f}")
    print(f"  Min/Max episode length: {min_length}/{max_length}")
    print(f"  Episodes with exactly {JOB_SEQUENCE_SIZE} steps: {exact_256_count}/{num_episodes}")
    print(f"  Success rate: {exact_256_count/num_episodes*100:.1f}%")
    
    if exact_256_count == num_episodes:
        print(f"  ✓ ALL episodes were exactly {JOB_SEQUENCE_SIZE} steps!")
    elif exact_256_count > 0:
        print(f"  ⚠ Some episodes were exactly {JOB_SEQUENCE_SIZE} steps")
    else:
        print(f"  ✗ NO episodes were exactly {JOB_SEQUENCE_SIZE} steps")

if __name__ == '__main__':
    print("HPCEnv Episode Length Test (Exact PPO Config)")
    print("=" * 60)
    test_hpcenv_episode_length_exact_config(num_episodes=5, seed=42)
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
