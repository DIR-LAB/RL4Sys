import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import numpy as np
import torch
import os
from datetime import datetime

# Set random seed
SEED = 0

# Create the environment
def make_env():
    env = gym.make("LunarLander-v3")
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    env.observation_space.seed(SEED)
    return env

# Wrapping environment for monitoring and vectorization
env = DummyVecEnv([lambda: Monitor(make_env())])

# Generate a unique TensorBoard log folder
script_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(script_dir, "logs", f"dqn_baseline3_{timestamp}")
os.makedirs(log_dir, exist_ok=True)

# Configure logger for TensorBoard
new_logger = configure(log_dir, ["stdout","tensorboard"])

# Hyperparameters for Double DQN
policy_kwargs = dict(
    net_arch=[64, 64],  # Two fully connected layers with 64 units each
)

# Initialize the Double DQN model
model = DQN(
    "MlpPolicy", 
    env, 
    learning_rate=1e-3,
    buffer_size=50000,  # Replay buffer size
    learning_starts=1,  # Start learning after this many steps
    batch_size=64,
    tau=1e-3,  # Soft update coefficient
    gamma=0.99,  # Discount factor
    train_freq=64,  # Train every 4 steps
    gradient_steps=1,  # Gradient updates per training step
    target_update_interval=500,  # Target network update frequency
    exploration_fraction=0.1,  # Fraction of steps for exploration
    exploration_initial_eps=1.0,  # Initial epsilon for exploration
    exploration_final_eps=0.01,  # Final epsilon for exploration
    policy_kwargs=policy_kwargs,  # Custom policy network
    verbose=1,
    seed=SEED,
    device=torch.device("cpu")  # Run on CPU
)

# Attach the custom logger to the model
model.set_logger(new_logger)

# Train the model
TIMESTEPS = 100_000
model.learn(total_timesteps=TIMESTEPS)

# Save the model
# model.save("dqn_lunarlander")

# Evaluate the trained model
env = make_env()
state, _ = env.reset(seed=SEED)
done = False
score = 0

while not done:
    action, _ = model.predict(state, deterministic=True)
    state, reward, done, _, _ = env.step(action)
    score += reward

print(f"Final Score: {score}")
env.close()
