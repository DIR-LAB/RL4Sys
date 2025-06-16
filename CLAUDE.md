# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

RL4Sys is a distributed reinforcement learning framework with a server-client architecture:

- **Server**: Manages multiple client training sessions, each with dedicated algorithm instances, training threads, and model version managers
- **Client**: Runs RL agents that communicate with the server via gRPC to share trajectories and receive model updates
- **Algorithms**: Supports multiple RL algorithms (PPO, DQN, SAC, DDPG, etc.) with configurable hyperparameters
- **Communication**: Uses Protocol Buffers and gRPC for efficient client-server communication

The framework follows a client-specific training approach where each client gets its own algorithm instance and training thread, coordinated by a central dispatcher.

## Development Commands

### Python Environment
```bash
# Install dependencies
pip install -e .

# Start the RL4Sys server
cd rl4sys
python start_server.py --debug

# Run the Lunar Lander example
cd rl4sys/examples/lunar
python lunar_lander.py --debug
```

### Protocol Buffers
```bash
# Generate gRPC Python stubs
cd rl4sys/proto
./generate_proto.sh
```

### C++ Client
```bash
# Build C++ client library
cd rl4sys/cppclient
mkdir build && cd build
cmake ..
make

# Run C++ tests
make test
# or
ctest
```

### Monitoring
```bash
# View training logs with TensorBoard
cd rl4sys/logs
tensorboard --logdir rl4sys-ppo-info
```

## Configuration System

- **config.json**: Global algorithm configurations and server settings
- **Client configs**: Each client uses a JSON config file (e.g., `luna_conf.json`) specifying:
  - Algorithm choice and hyperparameters
  - Network architecture parameters
  - Server address and communication settings
  - Environment-specific parameters

## Key Implementation Details

### Server Architecture (`rl4sys/server/`)
- `server.py`: Main gRPC server with client-specific training threads
- `model_diff_manager.py`: Handles model versioning and differential updates

### Client Implementation (`rl4sys/client/`)
- `agent.py`: Core RL4SysAgent class that manages training loops
- `config_loader.py`: Handles client configuration loading

### Algorithms (`rl4sys/algorithms/`)
Each algorithm directory contains:
- Main algorithm implementation (e.g., `PPO.py`)
- `kernel.py`: Core training logic
- `replay_buffer.py`: Experience replay management

### Common Types (`rl4sys/common/`)
- `action.py`: RL4SysAction class for action representation
- `trajectory.py`: RL4SysTrajectory class for experience collection

## Code Style Guidelines

### Python
- Use type annotations for all functions and classes
- Follow PEP 257 for docstrings
- Maintain existing comments and documentation style

### C++
- Use PascalCase for classes, camelCase for functions/variables
- Prefer smart pointers over raw pointers
- Use Doxygen-style comments for public APIs
- Follow RAII principles for resource management

## Dependencies

Core Python dependencies include:
- torch, numpy, scipy (ML/numerical computing)
- grpcio, grpcio-tools (communication)
- gymnasium, gym (RL environments)
- tensorboard (logging/monitoring)
- pygame, box2d (environment rendering)

C++ dependencies require:
- gRPC and Protocol Buffers
- GoogleTest (for testing)
- CMake 3.15+ (build system)