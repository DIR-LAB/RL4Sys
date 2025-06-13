from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium.spaces import Box, Discrete, Dict
from ray.rllib.policy.policy import PolicySpec

# https://github.com/ray-project/ray/issues/46087
# reference of using

obs_space = Box(float("-inf"), float("inf"), (8,))
action_space = Discrete(4)
config = (
    PPOConfig()
    .environment(
        env=None,
        observation_space=obs_space,
        action_space=action_space,
    )
    .framework("torch")
    .training(
        model={
            "fcnet_hiddens": [64, 64],  # Two hidden layers with 64 neurons each (8-64-64-4)
            "fcnet_activation": "relu",
        }
    )
    .offline_data(input_=lambda ioctx: PolicyServerInput(ioctx, "127.0.0.1", 1337))
    .env_runners(num_env_runners=0, enable_connectors=False)
    .debugging(log_level="DEBUG")
)

algo = config.build()

while True:
    results = algo.train()