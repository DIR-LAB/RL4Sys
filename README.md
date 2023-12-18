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

