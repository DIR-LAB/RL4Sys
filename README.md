# RL4Sys: Reinforcement Learning for System Control

RL4Sys is a distributed reinforcement learning framework designed for system control applications. It provides a server-client architecture that enables multiple clients to train and share models in a distributed manner.

## Project Structure

```
rl4sys/
├── algorithms/           # RL algorithm implementations
│   ├── PPO/             # Proximal Policy Optimization
│   └── DQN/             # Deep Q-Network
├── client/              # Client-side components
│   ├── agent.py         # RL agent implementation
│   └── config_loader.py # the configuration loader for client only
├── common/              # Shared utilities and components
|   |-- action.py        # Definition of RL4SysAction
|   |-- trajectory.py    # Definition of RL4SysTrajectory
├── examples/            # Example applications
│   └── lunar/          # Lunar Lander example
│       ├── lunar_lander.py
│       └── luna_conf.json
├── logs/               # Logging directory
├── proto/              # Protocol buffer definitions
|   |-- rl4sys.proto    # gRPC proto definition
|   |-- generate_proto.sh # script to generate the gRPC python stub code
├── server/             # Server-side components
│   ├── server.py       # Main server implementation
│   └── model_diff_manager.py  # Model versioning and diff management
├── utils/              # Utility functions
├── start_server.py     # Server startup script
└── __init__.py         # Package initialization
```

## Features

- Distributed training architecture with server-client model
- Support for multiple RL algorithms (PPO, DQN)
- Efficient model versioning and diff management
- Client-specific training threads
- Protocol buffer-based communication
- Comprehensive logging system
- Example implementations for system control tasks

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RL4Sys.git
cd RL4Sys
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Starting the Server

```bash
python -m rl4sys.start_server --debug --port 50051
```

### Running the Lunar Lander Example

The Lunar Lander example demonstrates how to use RL4Sys with the Gymnasium Lunar Lander environment:

```bash
python -m rl4sys.examples.lunar.lunar_lander.py \
    --seed 1 \
    --number-of-iterations 10000 \
    --number-of-moves 200 \
    --render False \
    --client-id lunar_lander
```

## Configuration

The Lunar Lander example uses a configuration file (`luna_conf.json`) to specify:
- Algorithm parameters
- Network architecture
- Training hyperparameters
- Communication settings

Example configuration:
```json
{
    "client_id": "luna-landing",
    "algorithm_name": "PPO",
    "algorithm_parameters": {
        "batch_size": 512,
        "act_dim": 4,
        "seed": 0,
        "traj_per_epoch": 256,
        "clip_ratio": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        "pi_lr": 3e-4,
        "vf_lr": 1e-3,
        "train_pi_iters": 80,
        "train_v_iters": 80,
        "target_kl": null,
        "input_size": 8
    },
    "act_limit": 1.0,
    "max_traj_length": 1000,
    "type": "onpolicy",
    "train_server_address": "localhost:50051",
    "send_frequency": 10
}
```

## How It Works

### Server Architecture

The RL4Sys server implements a client-specific training approach:

1. Each client gets its own:
   - Algorithm instance
   - Training thread
   - Model version manager
   - Training queue

2. A central dispatcher thread:
   - Receives trajectories from all clients
   - Routes them to the appropriate client's training queue

3. Client-specific training threads:
   - Process trajectories from their dedicated queue
   - Update their algorithm's model
   - Manage model versioning

### Client Implementation

The Lunar Lander example shows how to implement a client:

1. Initialize the RL4Sys agent:
```python
self.rlagent = RL4SysAgent(conf_path='./luna_conf.json')
```

2. Run the training loop:
```python
def run_application(self, num_iterations, max_moves):
    for iteration in range(num_iterations):
        obs, _ = self.env.reset(seed=self._seed + iteration)
        done = False
        moves = 0
        
        while not done and moves < max_moves:
            # Get action from agent
            self.rl4sys_traj, self.rl4sys_action = self.rlagent.request_for_action(
                self.rl4sys_traj, obs_tensor
            )
            
            # Execute action and get reward
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            
            # Update trajectory
            self.rlagent.add_to_trajectory(self.rl4sys_traj, self.rl4sys_action)
            self.rl4sys_action.update_reward(reward)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.