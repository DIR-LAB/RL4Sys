RL4SysAgent – Client‑side API Documentation
=============================================

*(Generated: 2025‑06‑23)*

RL4SysAgent is a lightweight client wrapper that lets any Python process
interact with an **RL4Sys** training server.  It speaks gRPC under the hood,
streams trajectories asynchronously, and periodically refreshes the local copy
of the policy network.

--------------------------------------------------------------------
Constructor
--------------------------------------------------------------------
```
RL4SysAgent(conf_path: str, debug: bool = False)
```

| Parameter  | Type | Description                                                             |
|------------|------|-------------------------------------------------------------------------|
| conf_path  | str  | Path to a JSON/YAML file containing **client‑side** configuration.      |
| debug      | bool | If *True*, enables verbose `StructuredLogger` output (default False).   |

The configuration file provides:

* **client_id** – free‑form string identifying this client instance.  
* **algorithm_name** – e.g. "PPO", "DQN" (must exist in `MODEL_CLASSES`).  
* **algorithm_type** – “onpolicy” or “offpolicy”; controls trajectory filtering.  
* **algorithm_parameters** – hyper‑parameters such as `input_size`, `act_dim`.  
* **train_server_address** – host:port of the RL4Sys gRPC service.  
* **send_frequency** – number of *completed* trajectories buffered before they
  are queued for upload.

The constructor:

1. Parses the file with `AgentConfigLoader`.  
2. Opens an insecure gRPC channel (`grpc.insecure_channel`).  
3. Sends an `InitAlgorithm` request to the server so the trainer knows which
   algorithm/hyper‑parameters to instantiate.  
4. Downloads the initial model weights and stores the version.  
5. Spawns a daemon thread that flushes completed trajectories in the background.

--------------------------------------------------------------------
Public Methods
--------------------------------------------------------------------

### request_for_action

```
request_for_action(
    traj: RL4SysTrajectory | None,
    obs:  torch.Tensor,
    *args, **kwargs
) -> tuple[RL4SysTrajectory, RL4SysAction]
```

Generate an action from the **current** policy.

* If `traj` is `None` or already completed, a fresh `RL4SysTrajectory`
  is created and placed in the internal buffer.
* The model’s `.step(obs)` method returns both the raw action tensor and
  any algorithm‑specific diagnostics; these are wrapped into an
  `RL4SysAction`.
* **Returns** the trajectory handle (for continued use) and the new action.

Thread‑safe: obtains `self._lock` while accessing the model.

---

### add_to_trajectory

```
add_to_trajectory(traj: RL4SysTrajectory,
                  action: RL4SysAction) -> None
```

Appends `action` to `traj`.  
Call this **once per env step**, immediately after `request_for_action`.

---

### mark_end_of_trajectory

```
mark_end_of_trajectory(traj: RL4SysTrajectory,
                       action: RL4SysAction) -> None
```

Flags `action.done = True`, marks `traj` as completed, and invokes an
internal check to see whether enough finished trajectories are buffered to
trigger an upload via the background thread.

---

### update_action_reward

```
update_action_reward(action: RL4SysAction,
                     reward: float) -> None
```

Sets `action.reward = reward`.  Must be done *before* the next
`request_for_action` so that the server receives a fully‑formed
(observation, action, reward) triple.

---

### close

```
close() -> None
```

* Signals the sending thread to stop (`self._stop_event`).  
* Joins the thread.  
* Closes the gRPC channel.

Always call this from your application’s shutdown path to avoid hanging
daemon threads.

--------------------------------------------------------------------
Important Attributes
--------------------------------------------------------------------
| Attribute                 | Type/Meaning                                    |
|---------------------------|-------------------------------------------------|
| `client_id`               | String identifier of this agent instance.       |
| `algorithm_name`          | Name of the RL algorithm (“PPO”, “DQN”…).       |
| `local_version`           | Currently loaded model version (int).           |
| `_model`                  | Torch `nn.Module` implementing the policy.      |
| `_trajectory_buffer`      | List[RL4SysTrajectory]; pending & running.      |
| `_send_queue`             | `queue.Queue` used by the background thread.    |
| `_lock`                   | `threading.Lock` guarding model access.         |
| `_trajectory_lock`        | `threading.Lock` guarding trajectory buffer.    |

--------------------------------------------------------------------
Execution Flow Summary
--------------------------------------------------------------------
1. **Env step loop** calls  
   `traj, action = request_for_action(traj, obs)`  
   `add_to_trajectory(traj, action)`  
   `env.step(action.act)`  
   `action.update_reward(reward)`

2. On episode completion → `mark_end_of_trajectory(traj, action)`.

3. Background thread drains completed trajectories when their count
   reaches `send_frequency`, calls `SendTrajectories`, and—if the response
   indicates an updated model—fetches the new weights with `GetModel`.

--------------------------------------------------------------------
Minimal Usage Example
--------------------------------------------------------------------
```python
agent = RL4SysAgent("luna_conf.json")
env   = gym.make("LunarLander-v3")

obs, _ = env.reset(seed=0)
traj   = None
done   = False
while not done:
    traj, act = agent.request_for_action(traj, torch.tensor(obs).unsqueeze(0))
    agent.add_to_trajectory(traj, act)

    obs, reward, terminated, truncated, _ = env.step(int(act.act))
    act.update_reward(reward)

    if terminated or truncated:
        agent.mark_end_of_trajectory(traj, act)
        done = True

agent.close()
```
