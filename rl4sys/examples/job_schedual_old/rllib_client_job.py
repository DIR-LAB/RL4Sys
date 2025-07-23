import os
import sys

# ------------------------------------------------------------------
#  Ensure rl4sys package & HPCSim module resolvable before further imports
# ------------------------------------------------------------------
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

hpcsim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'HPCSim'))
if hpcsim_path not in sys.path:
    sys.path.insert(0, hpcsim_path)

import time
from typing import Dict, Union
from ray.rllib.env.policy_client import PolicyClient
import gymnasium as gym
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from rl4sys.utils.mem_prof import MemoryProfiler
from rl4sys.utils.cpu_prof import CPUProfiler
from rl4sys.utils.step_per_sec_log import StepPerSecLogger


# HPCSim job scheduling environment and constants
from rl4sys.examples.job_schedual_old.HPCSim.HPCSimPickJobs import HPCEnv, JOB_FEATURES, MAX_QUEUE_SIZE
from rl4sys.utils.step_statistic import compute_field_statistics

# -------------------------------------------------------------
#  Utility: build binary action mask from HPCSim observation
# -------------------------------------------------------------
global mode
mode = "local"

def compute_action_mask(obs: np.ndarray) -> np.ndarray:
    """Compute binary mask over job slots.

    A slot is invalid (mask = 0) if it is empty or filled w/ sentinel patterns
    as defined in JobSchedulingSim.build_mask (empty: [0, 1,1,1,1,1,1,0],
    filled: [1,1,1,1,1,1,1,1]). Otherwise valid (mask = 1).
    """

    mask = np.ones(MAX_QUEUE_SIZE, dtype=np.float32)
    for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
        job_slot = obs[i : i + JOB_FEATURES]
        idx = i // JOB_FEATURES

        # Pattern checks use rounding because obs is float.
        slot_int = np.round(job_slot).astype(int)
        if np.array_equal(slot_int, np.array([0] + [1] * (JOB_FEATURES - 2) + [0])):
            mask[idx] = 0.0  # Empty slot
        elif np.array_equal(slot_int, np.ones(JOB_FEATURES, dtype=int)):
            mask[idx] = 0.0  # Filled slot (already scheduled)

    return mask

# -------------------------------------------------------------
#  Register same custom kernel+mask model for local inference
# -------------------------------------------------------------


from rl4sys.algorithms.PPO.kernel import RLActor, RLCritic


class RLKernelMaskModel(TorchModelV2, nn.Module):
    """Same model as server side (kernel network with mask)."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if hasattr(obs_space, "spaces"):
            self.input_size = obs_space.spaces["obs"].shape[0]
        else:
            self.input_size = obs_space.shape[0] - MAX_QUEUE_SIZE

        self.act_dim = action_space.n

        self.actor = RLActor(self.input_size, self.act_dim, actor_type="kernel", job_features=JOB_FEATURES)
        self.critic = RLCritic(self.input_size)
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        d = input_dict["obs"]
        obs_flat = d["obs"].float() if isinstance(d, dict) else d.float()
        mask = d.get("action_mask", None)
        if mask is not None:
            mask = mask.float()

        dist = self.actor._distribution(obs_flat, mask)
        logits = dist.logits
        self._value_out = self.critic(obs_flat)
        return logits, state

    def value_function(self):
        return self._value_out


# Register (if duplicate name, Ray's registry will overwrite or ignore)
try:
    ModelCatalog.register_custom_model("rl_kernel_mask_model", RLKernelMaskModel)
except Exception:
    pass


def run_rllib_client(num_steps: int = 4000) -> Dict[str, Union[float, int]]:
    """
    Run RLlib client with profiling to measure performance metrics.
    
    Args:
        num_steps: Number of environment steps to run for profiling
    """
    # Initialize HPCSim environment
    env = HPCEnv()

    # Locate default workload file (lublin_256.swf)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    workload_path = os.path.join(
        root_dir,
        'rl4sys',
        'examples',
        'job_schedual_old',
        'HPCSim',
        'data',
        'lublin_256.swf',
    )

    # Load workload and seed environment
    env.my_init(workload_file=workload_path, sched_file='')
    env.seed(0)
    # Use remote inference mode to rely on the server-side custom model.
    #mode = "remote"
    
    client = PolicyClient("http://127.0.0.1:1337", inference_mode=mode)
    
    # ------------------------------------------------------
    # TensorBoard logger setup
    # ------------------------------------------------------
    log_dir = os.path.join("rlliblog", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Initialize timing variables
    env_time = 0.0
    infer_time = 0.0
    log_time = 0.0
    step_count = 0
    t0_total = time.perf_counter()
    
    obs, _ = env.reset()
    action_mask = compute_action_mask(obs)
    obs_dict = {"obs": obs, "action_mask": action_mask}
    eid = client.start_episode()
    
    terminated, truncated = False, False

    episode_count = 0

    metric_lst = []
    step_logger = StepPerSecLogger("rllib_client_job_" + mode)
    
    while episode_count < 100:
        # Update action mask & build obs dict
        action_mask = compute_action_mask(obs)
        obs_dict = {"obs": obs, "action_mask": action_mask}

        # Time inference (getting action from RLlib server)
        t0 = time.perf_counter()
        action = client.get_action(eid, obs_dict)    # client poll model every 10 seconds
        infer_time += time.perf_counter() - t0  # episode_id to sync training: https://github.com/ray-project/ray/blob/master/rllib/evaluation/collectors/simple_list_collector.py?utm_source=chatgpt.com line 419
        # weight transfer: https://ray-project.github.io/q4-2021-docs-hackathon/0.4/ray-api-references/ray-rllib/evaluation/#ray.rllib.evaluation.rollout_worker.RolloutWorker.get_weights
        # search: list(self.policy_map.keys()) in above link
        # weight transfer policy layer: https://github.com/ray-project/ray/blob/master/rllib/policy/torch_policy.py line 713 (V1 version, not V2)
        # We confirm weight transfer is: {k: v.cpu().detach().numpy() for k, v in self.model.state_dict().items()}
        
        # Time environment step within HPCSim environment
        t0 = time.perf_counter()
        step_result = env.step(action)
        obs, reward, done_flag, reward2, sjf_t, f1_t = step_result
        terminated, truncated = bool(done_flag), False
        info = {"reward2": reward2, "sjf_t": sjf_t, "f1_t": f1_t}
        env_time += time.perf_counter() - t0

        
        
        t0 = time.perf_counter() 
        client.log_returns(eid, reward, info=info)
        log_time += time.perf_counter() - t0
        
        step_count += 1
    
        if terminated or truncated:
            print("episode_count: ", episode_count, "step_count: ", step_count, "reward: ", reward, "reward2: ", reward2, "sjf_t: ", sjf_t, "f1_t: ", f1_t)
            # Log reward to TensorBoard
            writer.add_scalar("reward", reward, episode_count)
            writer.add_scalar("reward2", reward2, episode_count)
            writer.add_scalar("sjf_t", sjf_t, episode_count)
            writer.add_scalar("f1_t", f1_t, episode_count)
            episode_count += 1
            obs, _ = env.reset()
            action_mask = compute_action_mask(obs)
            obs_dict = {"obs": obs, "action_mask": action_mask}
            eid = client.start_episode()
            terminated, truncated = False, False

            # Calculate and print performance metrics
            total_time = time.perf_counter() - t0_total
            per_step_ms = total_time * 1000 / step_count
            env_ms = env_time * 1000 / step_count
            infer_ms = infer_time * 1000 / step_count
            log_ms = log_time * 1000 / step_count
            over_ms = per_step_ms - env_ms - infer_ms - log_ms
            
            metrics = {
                "steps/s": round(1000 / per_step_ms, 1),
                "env_ms": round(env_ms, 3),
                "infer_ms": round(infer_ms, 3),
                "log_ms": round(log_time, 3),
                "over_ms": round(over_ms, 3),
            }
            print(metrics)

            metric_lst.append(metrics) # append metrics to list
            step_logger.log(metrics["steps/s"])

    for i in metric_lst:
        print(i)
    # Flush and close TensorBoard writer
    writer.flush()
    writer.close()
    step_logger.close()

    print(compute_field_statistics(metric_lst)) 
    return metric_lst


if __name__ == "__main__":
    memory_profiler = MemoryProfiler("rllib_client_job_" + mode, log_interval=0.5)
    memory_profiler.start_background_profiling()
    cpu_profiler = CPUProfiler("rllib_client_job_" + mode, log_interval=0.5)
    cpu_profiler.start_background_profiling()

    metric_lst = []
    for i in range(1):
        metric_lst.append(run_rllib_client())
        
    memory_profiler.stop_background_profiling()
    cpu_profiler.stop_background_profiling()
