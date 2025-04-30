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
│   └── ...
├── proto/               # Protocol buffer definitions
├── server/              # Server-side components
│   ├── server.py        # Main server implementation
│   └── model_diff_manager.py  # Model versioning and diff management
├── utils/               # Utility functions
└── examples/            # Example applications
    └── lunar/           # Lunar Lander example
        ├── lunar_lander.py
        └── luna_conf.json
```

## Features

- Distributed training architecture
- Support for multiple RL algorithms (PPO, DQN)
- Efficient model versioning and diff management
- Client-specific training threads
- Protocol buffer-based communication
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
python -m rl4sys.server.server --debug --port 50051
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
    "algorithm": "PPO",
    "algorithm_parameters": {
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "target_kl": 0.01,
        "entropy_coef": 0.01
    },
    "network_architecture": {
        "hidden_sizes": [64, 64],
        "activation": "tanh"
    },
    "training": {
        "batch_size": 64,
        "epochs": 10
    },
    "communication": {
        "server_address": "localhost:50051",
        "timeout": 10
    }
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