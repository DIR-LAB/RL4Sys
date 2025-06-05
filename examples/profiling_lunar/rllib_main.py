#!/usr/bin/env python
import time, gymnasium as gym, ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# ① 计时包裹 Env（跟之前一致）
class TimedEnv(gym.Env):
    def __init__(self, _): self.env = gym.make("LunarLander-v3")
    def reset(self, **k):   return self.env.reset(**k)
    def step(self, a):
        t0 = time.perf_counter()
        o, r, d, t, info = self.env.step(a)
        info["step_time"] = time.perf_counter() - t0
        return o, r, d, t, info
    @property
    def observation_space(self): return self.env.observation_space
    @property
    def action_space(self):      return self.env.action_space

# ② 新版字段→overhead
class OverheadCB(DefaultCallbacks):
    def on_train_result(self, *, result, **kw):
        data = result.get("env_runners", {})
        if not data:   # 首轮热身或字段缺失
            return
        per_step_ms = (data["sample"] / data["num_env_steps_sampled"]) * 1000
        env_ms      = data["env_step_timer"]           * 1000
        infer_ms    = data["rlmodule_inference_timer"] * 1000
        overhead    = per_step_ms - env_ms - infer_ms
        result.setdefault("custom_metrics", {})["overhead_ms_per_step"] = overhead

# ③ 配置
cfg = (
    PPOConfig()
    .environment(TimedEnv)
    .env_runners(num_env_runners=0)    # 本地单进程
    .framework("torch")
    .training(model={
        "fcnet_hiddens": [64, 64],     # ← 手动对齐
        "fcnet_activation": "relu",
    })
    .training(train_batch_size=4000, minibatch_size=1024, num_epochs=1)
    .evaluation(evaluation_duration=1, evaluation_duration_unit="episodes")
    .callbacks(OverheadCB)
)

ray.init(local_mode=False)
algo = cfg.build()

# ④ 训练几轮并打印
for i in range(5):
    r    = algo.train()
    data = r["env_runners"]
    per_step_ms = (data["sample"] / data["num_env_steps_sampled"]) * 1000
    env_ms      = data["env_step_timer"]           * 1000
    infer_ms    = data["rlmodule_inference_timer"] * 1000
    over_ms     = r["custom_metrics"]["overhead_ms_per_step"]
    print({
        "iter": i,
        "steps/s": round(1000 / per_step_ms, 1),
        "env_ms":  round(env_ms,    3),
        "infer_ms":round(infer_ms,  3),
        "over_ms": round(over_ms,   3),
    })

ray.shutdown()
