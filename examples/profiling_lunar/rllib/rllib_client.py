import time
from typing import Dict, Union
from ray.rllib.env.policy_client import PolicyClient
from ray.rllib.examples.envs.classes.multi_agent import make_multi_agent
import gymnasium as gym


def run_rllib_client(num_steps: int = 4000) -> None:
    """
    Run RLlib client with profiling to measure performance metrics.
    
    Args:
        num_steps: Number of environment steps to run for profiling
    """
    env = gym.make("LunarLander-v2")
    #client = PolicyClient("http://127.0.0.1:1337", inference_mode="remote")
    client = PolicyClient("http://127.0.0.1:1337", inference_mode="local")
    
    # Initialize timing variables
    env_time = 0.0
    infer_time = 0.0
    log_time = 0.0
    step_count = 0
    t0_total = time.perf_counter()
    
    obs, info = env.reset(seed=0)
    eid = client.start_episode()
    
    terminated, truncated = False, False
    
    while step_count < num_steps:
        if terminated or truncated:
            obs, info = env.reset()
            eid = client.start_episode()
            terminated, truncated = False, False
        
        # Time inference (getting action from RLlib server)
        t0 = time.perf_counter()
        action = client.get_action(eid, obs)    # client poll model every 10 seconds: https://github.com/ray-project/ray/blob/master/rllib/env/policy_client.py#L276
        infer_time += time.perf_counter() - t0  # episode_id to sync training: https://github.com/ray-project/ray/blob/master/rllib/evaluation/collectors/simple_list_collector.py?utm_source=chatgpt.com line 419
        # weight transfer: https://ray-project.github.io/q4-2021-docs-hackathon/0.4/ray-api-references/ray-rllib/evaluation/#ray.rllib.evaluation.rollout_worker.RolloutWorker.get_weights
        # search: list(self.policy_map.keys()) in above link
        # weight transfer policy layer: https://github.com/ray-project/ray/blob/master/rllib/policy/torch_policy.py line 713 (V1 version, not V2)
        # We confirm weight transfer is: {k: v.cpu().detach().numpy() for k, v in self.model.state_dict().items()}
        
        # Time environment step
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        env_time += time.perf_counter() - t0
        
        t0 = time.perf_counter()
        client.log_returns(eid, reward, info=info)
        log_time += time.perf_counter() - t0
        
        step_count += 1
    
    # Calculate and print performance metrics
    total_time = time.perf_counter() - t0_total
    per_step_ms = total_time * 1000 / num_steps
    env_ms = env_time * 1000 / num_steps
    infer_ms = infer_time * 1000 / num_steps
    log_ms = log_time * 1000 / num_steps
    over_ms = per_step_ms - env_ms - infer_ms - log_ms
    
    metrics = {
        "steps/s": round(1000 / per_step_ms, 1),
        "env_ms": round(env_ms, 3),
        "infer_ms": round(infer_ms, 3),
        "log_ms": round(log_time, 3),
        "over_ms": round(over_ms, 3),
    }

    return metrics


if __name__ == "__main__":
    metric_lst = []
    for i in range(5):
        metric_lst.append(run_rllib_client())

    for metric in metric_lst:
        print(metric)