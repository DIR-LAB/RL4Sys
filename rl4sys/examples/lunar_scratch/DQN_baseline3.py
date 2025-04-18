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
log_dir = os.path.join("runs", f"dqn_baseline3_{timestamp}")
os.makedirs(log_dir, exist_ok=True)

# Configure logger for TensorBoard
new_logger = configure(log_dir, ["stdout","tensorboard"])

# Hyperparameters from cleanRL
policy_kwargs = dict(
    net_arch=[120, 84],  # Match cleanRL's network architecture
)

# Initialize the Double DQN model
model = DQN(
    "MlpPolicy", 
    env, 
    learning_rate=2.5e-4,  # Match cleanRL's learning rate
    buffer_size=10000,     # Match cleanRL's buffer size
    learning_starts=10000, # Match cleanRL's learning starts
    batch_size=128,       # Match cleanRL's batch size
    tau=1.0,             # Match cleanRL's tau
    gamma=0.99,          # Same as cleanRL
    train_freq=10,       # Match cleanRL's train frequency
    gradient_steps=1,    # cleanRL does one update per training step
    target_update_interval=500,  # Match cleanRL's target network frequency
    exploration_fraction=0.5,    # Match cleanRL's exploration fraction
    exploration_initial_eps=1.0, # Match cleanRL's start epsilon
    exploration_final_eps=0.05,  # Match cleanRL's end epsilon
    policy_kwargs=policy_kwargs,
    verbose=1,
    seed=SEED,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Match cleanRL's device logic
)

# Attach the custom logger to the model
model.set_logger(new_logger)

# Train the model
TIMESTEPS = 500_000
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
