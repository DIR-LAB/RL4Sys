# policy_server.py
import ray, gymnasium as gym, torch, torch.nn as nn
from torchrl.modules import MLP
from pathlib import Path
import time
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class Observation(BaseModel):
    obs: list[float]

class Action(BaseModel):
    action: int

@ray.remote
class LearnerServer:
    def __init__(self, obs_dim, act_dim, lr=1e-3):
        # MLP(in_features, out_features, depth, num_cells, activation_class)
        self.policy = MLP(
            in_features=obs_dim,
            out_features=act_dim,
            depth=2,  # number of hidden layers
            num_cells=64,  # number of neurons in each hidden layer
            activation_class=nn.ReLU
        )
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = []          # 仅示例，生产用请换 ReplayBuffer

    def compute_action(self, obs):
        with torch.no_grad():
            logits = self.policy(torch.as_tensor(obs).float())
            return int(torch.distributions.Categorical(logits=logits).sample())

    def report_transition(self, obs, act, rew, nxt, done):
        self.buffer.append((obs, act, rew, nxt, done))
        if len(self.buffer) >= 1024:    # 每攒 1024 步简单更新一次
            obs, act, rew = zip(*[(o,a,r) for (o,a,r,_,_) in self.buffer])
            logits = self.policy(torch.as_tensor(obs).float())
            logp   = torch.log_softmax(logits, -1)
            act    = torch.as_tensor(act)
            loss   = - (logp[range(len(act)), act] * torch.as_tensor(rew)).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.buffer.clear()

# Initialize Ray and create server
ray.init()
server = LearnerServer.remote(obs_dim=8, act_dim=4)

@app.post("/action")
async def get_action(obs: Observation) -> Action:
    try:
        action = ray.get(server.compute_action.remote(obs.obs))
        return Action(action=action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_returns")
async def log_returns(obs: Observation, action: int, reward: float, done: bool):
    try:
        ray.get(server.report_transition.remote(obs.obs, action, reward, None, done))
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting TorchRL server on http://127.0.0.1:1337")
    uvicorn.run(app, host="127.0.0.1", port=1337)
