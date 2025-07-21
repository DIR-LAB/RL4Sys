from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium.spaces import Box, Discrete, Dict
from ray.rllib.policy.policy import PolicySpec
import numpy as np
import sys
import os

# ------------------------------------------------------------------
#  Ensure project root & HPCSim directories are on PYTHONPATH *first*
# ------------------------------------------------------------------

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, root_path)

hpcsim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'HPCSim'))
sys.path.insert(0, hpcsim_path)

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog

# Now that paths are set, import kernel networks
from rl4sys.algorithms.PPO.kernel import RLActor, RLCritic
import torch.nn as nn
import torch

# Import HPCSim constants for observation and action spaces
from rl4sys.examples.job_schedual_old.HPCSim.HPCSimPickJobs import JOB_FEATURES, MAX_QUEUE_SIZE
# -------------------------------------------------------------
#  Custom RLlib Model using "kernel" network and action masking
# -------------------------------------------------------------


class RLKernelMaskModel(TorchModelV2, nn.Module):
    """RLlib compatible model wrapping RLActor (kernel) and RLCritic.

    The model expects observations to be provided as a dict with two keys:
        "obs":          flattened observation vector (float32)
        "action_mask":  binary mask (1 = valid, 0 = invalid) of shape (MAX_QUEUE_SIZE,)
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Handle cases depending on whether preprocessor API is disabled (Dict) or not (flattened Box)
        if isinstance(obs_space, Dict):
            self.input_size = obs_space.spaces["obs"].shape[0]
        else:
            # Flattened Box: subtract mask length to get observation size
            self.input_size = obs_space.shape[0] - MAX_QUEUE_SIZE
        self.act_dim: int = action_space.n

        # Actor-Critic sub-networks
        self.actor: RLActor = RLActor(self.input_size, self.act_dim, actor_type="kernel", job_features=JOB_FEATURES)
        self.critic: RLCritic = RLCritic(self.input_size)

        # Placeholder for last value output required by RLlib
        self._value_out: torch.Tensor | None = None

    def forward(self, input_dict, state, seq_lens):
        obs_dict = input_dict["obs"]

        # Extract observation and mask
        if isinstance(obs_dict, dict):
            obs_flat = obs_dict["obs"].float()
            mask = obs_dict.get("action_mask", None)
            if mask is not None:
                mask = mask.float()
        else:
            obs_flat = obs_dict.float()
            mask = None

        # Distribution & logits
        dist = self.actor._distribution(obs_flat, mask)
        logits = dist.logits  # (batch, act_dim)

        # Store value for value_function()
        self._value_out = self.critic(obs_flat)

        return logits, state

    def value_function(self):
        # RLlib expects a 1-D tensor of shape (batch_size,). Avoid squeezing into 0-D.
        return self._value_out


# Register the custom model with RLlib
ModelCatalog.register_custom_model("rl_kernel_mask_model", RLKernelMaskModel)

# Add project root to PYTHON path to access HPCSim modules
# root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
# sys.path.insert(0, root_path)

# Append HPCSim directory to sys.path so its internal absolute imports (job, cluster) work.
# hpcsim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'HPCSim'))
# sys.path.insert(0, hpcsim_path)



# https://github.com/ray-project/ray/issues/46087
# reference of using

# Observation definitions (dict with mask)
flat_obs_space = Box(low=0.0, high=1.0, shape=(JOB_FEATURES * MAX_QUEUE_SIZE,), dtype=np.float32)
mask_space = Box(low=0.0, high=1.0, shape=(MAX_QUEUE_SIZE,), dtype=np.float32)
obs_space = Dict({
    "obs": flat_obs_space,
    "action_mask": mask_space,
})

# Action space remains discrete over job-slots
action_space = Discrete(MAX_QUEUE_SIZE)
config = (
    PPOConfig()
    .environment(
        env=None,
        observation_space=obs_space,
        action_space=action_space,
    )
    .framework("torch")
    .experimental(_disable_preprocessor_api=True)
    .training(
        model={
            "custom_model": "rl_kernel_mask_model",
            # Disable RLlib’s default FC-net entirely – our custom model provides logits & value.
        }
    )
    .offline_data(input_=lambda ioctx: PolicyServerInput(ioctx, "127.0.0.1", 1337))
    .env_runners(num_env_runners=0, enable_connectors=False)
    .debugging(log_level="DEBUG")
)

algo = config.build()
#print(algo.get_config().rollout_fragment_length) # auto
while True:
    results = algo.train()