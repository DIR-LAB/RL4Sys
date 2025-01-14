import gymnasium as gym

env = gym.make('LunarLander-v3')

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

print(obs_dim, act_dim)

SEED = 0
obs, _ = env.reset(seed=SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

print(obs)
for i in range(10):
    obs, _, _ ,_ ,_  = env.step(1)


