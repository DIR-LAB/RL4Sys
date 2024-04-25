TODO: bring over plot.py, DQN, DQN soft updates, all #TODO comments, make sure all parameters can be changed (e.g. gamma, lambda for replay buffer), PEP 8 line width, check docstrings for consistency (capitalization, punctuation)

user-definables:
* environment script, using RL4SysAgent
    * should build observations, then interact with trajectory and model through RL4SysAgent
    * option to start instance of training_server if __name__==main
    * should specify parameters for training server
* algorithm
    * kernel(kernel_dim: int, kernel_size: int):
        step(flattened_obs: Tensor, mask: Tensor) -> ndarray, dict
        act(flattened_obs: Tensor, mask: Tensor) -> ndarray, dict
    * algorithm(kernel_dim: int, kernel_size: int, seed: int, *hyperparams):
        traj, epoch: int
        save(filename: str) -> None
        receive_trajectory(trajectory: RL4SysTrajectory) -> None
* ~~RL4SysObservation~~ removed from implementation for version 0.2.0

# RL4Sys

## Agent APIs

The loop of using RL4Sys agent

1. First, users implement observation.
2. Second, call action() to get output of 

a1 = agent.action(obv)
a1 = agent.action_with_mask(obv)

`action()` returns a RL4SysAction instance and users need to interpret it.

* a1.action()
* a1.prob()
* a1.logprob()
* a1.reward() to report the rewards.

3. Third, store the interaction into Reply Buffer.


4. Third, determine if a trajectory is done based on the systems' own logic.


RL4SysReplayBuffer() instance
buff.store(RLSysAction)

if traj is done
    buff.finish_traj(RLSysAction)

if done:
    reset()