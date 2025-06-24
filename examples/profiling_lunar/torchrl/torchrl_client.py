# env_client.py
import ray, gymnasium as gym, uuid, torch
from pathlib import Path
import time
import requests
import numpy as np
from typing import Dict
ray.init()

# 读取服务器句柄
handle = bytes.fromhex(Path("server_handle.txt").read_text().strip())
server = ray.get_actor(handle)

env = gym.make("LunarLander-v3")
env.reset(seed=0)

def run_env(seed=0, episodes=5):
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        act = ray.get(server.compute_action.remote(obs))
        nxt, rew, term, trunc, _ = env.step(act)
        done = term or trunc
        ray.get(server.report_transition.remote(obs, act, rew, nxt, done))
        obs = nxt
    

def run_torchrl_client(num_steps: int = 4000) -> Dict:
    """
    Run TorchRL client with profiling to measure performance metrics.
    
    Args:
        num_steps: Number of environment steps to run for profiling
    """
    env = gym.make("LunarLander-v2")
    server_url = "http://127.0.0.1:1337"
    
    # Initialize timing variables
    env_time = 0.0
    infer_time = 0.0
    log_time = 0.0
    step_count = 0
    t0_total = time.perf_counter()
    
    obs, info = env.reset(seed=0)
    
    terminated, truncated = False, False
    
    while step_count < num_steps:
        if terminated or truncated:
            obs, info = env.reset()
            terminated, truncated = False, False
        
        # Time inference (getting action from server)
        t0 = time.perf_counter()
        response = requests.post(f"{server_url}/action", json={"obs": obs.tolist()})
        action = response.json()["action"]
        infer_time += time.perf_counter() - t0
        
        # Time environment step
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        env_time += time.perf_counter() - t0
        
        # Time logging returns
        t0 = time.perf_counter()
        requests.post(
            f"{server_url}/log_returns",
            json={
                "obs": obs.tolist(),
                "action": int(action),
                "reward": float(reward),
                "done": bool(terminated or truncated)
            }
        )
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
        "log_ms": round(log_ms, 3),
        "over_ms": round(over_ms, 3),
    }
    
    return metrics

# 启动 N 个异步环境
if __name__ == "__main__":
    for i in range(4):
        run_env()
    env.close()

    metric_lst = []
    for i in range(5):
        metric_lst.append(run_torchrl_client())
    
    for metric in metric_lst:
        print(metric)