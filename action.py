class RL4SysAction:
    def __init__(self, obs, cobs, action, mask, reward, v_t, logp_t, done):
        self.obs = obs
        self.cobs = cobs
        self.act = action
        self.mask = mask
        self.rew = reward
        self.val = v_t
        self.logp = logp_t
        self.done = done

    def update_reward(self, reward):
        self.rew = reward
