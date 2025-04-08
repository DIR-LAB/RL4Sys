import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from server.training_server import start_training_server

# Constants matching lunar_lander.py
INPUT_DIM = 8
ACT_DIM = 4
MOVE_SEQUENCE_SIZE = 500

def start_lunar_server(algorithm_name='DQN', seed=1, tensorboard=False, extras=None):
    """
    Starts a training server for Lunar Lander.
    
    Args:
        algorithm_name (str): Name of the RL algorithm to use
        seed (int): Random seed for reproducibility
        tensorboard (bool): Whether to enable tensorboard logging
        extras (list): Additional arguments to pass to the training server
    """
    if extras is None:
        extras = []
    
    # Add buffer size parameter if not already present
    if '--buf_size' not in extras:
        extras.append('--buf_size')
        extras.append(str(MOVE_SEQUENCE_SIZE * 100))

    print(f"[lunar_server.py] Starting {algorithm_name} training server...")
    
    try:
        # Start the training server directly in the current thread
        # This call is blocking and will run until the server is terminated
        start_training_server(
            algorithm_name=algorithm_name,
            input_size=INPUT_DIM,
            action_dim=ACT_DIM,
            hyperparams=extras,
            env_dir=os.path.dirname(os.path.abspath(__file__)),
            tensorboard=tensorboard
        )
    except Exception as e:
        print(f"Error starting training server: {e}")
        print("If TensorBoard is enabled, try running with --tensorboard=False")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="RL4Sys Lunar Lander Training Server",
                                    formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--algorithm', '-a', type=str, default='PPO',
                        help='RL algorithm to use (DQN, PPO, DDPG, etc.)')
    parser.add_argument('--tensorboard', '-t', action='store_true', default=False,
                        help='enable tensorboard logging for training observations')
    parser.add_argument('--seed', '-s', type=int, default=1,
                        help='seed for random number generation')
    parser.add_argument('--buf_size', type=int, default=MOVE_SEQUENCE_SIZE * 100,
                        help='replay buffer size')
    
    args, extras = parser.parse_known_args()
    
    # Add buffer size to extras if specified in args
    extras.append('--buf_size')
    extras.append(str(args.buf_size))
    
    # Start the server directly
    start_lunar_server(
        algorithm_name=args.algorithm,
        seed=args.seed,
        tensorboard=args.tensorboard,
        extras=extras
    )
