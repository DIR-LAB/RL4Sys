import time, gymnasium as gym
class StepTimer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env); self.env_time = 0; self.n = 0
    def step(self, action):
        t0 = time.perf_counter_ns()
        obs, r, d, t, info = self.env.step(action)
        self.env_time += time.perf_counter_ns() - t0; self.n += 1
        return obs, r, d, t, info