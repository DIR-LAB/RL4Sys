import ray
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.env.external_env import ExternalEnv
import numpy as np


ray.init()
from ray.rllib.env.external_env import ExternalEnv
from ray.tune.registry import register_env
import gymnasium as gym
import numpy as np

class MyExternalEnv(ExternalEnv):
    def __init__(self, config):
        super().__init__(
            observation_space=gym.spaces.Box(-np.inf, np.inf, (8,), np.float32),
            action_space=gym.spaces.Discrete(4),
        )
    def run(self):            # 必须实现!
        while True:
            # 从外部系统/Socket读数据
            obs = self.get_data_from_sim()
            eid = self.start_episode()
            #self.log_returns(eid, reward, done)

register_env("ext_env", lambda cfg: MyExternalEnv(cfg))

config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .environment(env="ext_env")        # 传字符串或类；都可 Pickle
    .framework("torch")
    .offline_data(input_=lambda io: PolicyServerInput(io, "0.0.0.0", 9900))
    .env_runners(num_env_runners=0, enable_connectors=False)
)
algo = config.build_algo()
