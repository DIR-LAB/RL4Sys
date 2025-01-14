import gymnasium as gym

env = gym.make('LunarLander-v3')

SEED = 0
obs, _ = env.reset(seed=SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

print(obs)
for i in range(10):
    obs, _, _ ,_ ,_  = env.step(1)


