import ray
import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.policy_server_input import PolicyServerInput
import numpy as np


ray.init()

class LunarLanderStub(gym.Env):
    """仅用来向 RLlib 报告空间的占位环境，不会真正被 reset/step。"""
    def __init__(self, config=None):
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)  

def _input(ioctx):
        # We are remote worker or we are local worker with num_env_runners=0:
        # Create a PolicyServerInput.
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                "localhost",
                9900,
            )
        # No InputReader (PolicyServerInput) needed.
        else:
            return None
config = (
        PPOConfig()
        # Indicate that the Algorithm we setup here doesn't need an actual env.
        # Allow spaces to be determined by user (see below).
        .environment(
            env=None,
            observation_space=gym.spaces.Box(float("-inf"), float("inf"), (8,)),
            action_space=gym.spaces.Discrete(4),
        )
        # DL framework to use.
        .framework("torch")
        # Use the `PolicyServerInput` to generate experiences.
        .offline_data(input_=lambda io: PolicyServerInput(io, "localhost", 9900))
        # Use n worker processes to listen on different ports.
        .env_runners(
            num_env_runners=0,
            # Connectors are not compatible with the external env.
            enable_connectors=False,
        )
        # Disable OPE, since the rollouts are coming from online clients.
        .evaluation(off_policy_estimation_methods={})
        # Set to INFO so we'll see the server's actual address:port.
        .debugging(log_level="INFO")
    )
config.update_from_dict(
            {
                "rollout_fragment_length": 1000,
                "train_batch_size": 4000,
                "model": {"use_lstm": True},
            }
        )
algo = config.build()
for _ in range(5):
    results = algo.train()
    print(results)

algo.stop()