from stable_baselines3.common.callbacks import BaseCallback
import time, numpy as np

class PerfCallback(BaseCallback):
    def _on_training_start(self):               # reset counters
        self.t_env = self.t_infer = self.t_tot = 0; self.steps = 0
    def _on_rollout_start(self):
        self.rollout_t0 = time.perf_counter_ns()
    def _on_step(self) -> bool:
        self.steps += 1
        return True
    def _on_rollout_end(self):
        self.t_tot += time.perf_counter_ns() - self.rollout_t0
    def _on_predict(self, *args, **kwargs):     # monkey-patched by callback_init
        t0 = time.perf_counter_ns()
        actions = self.model.policy_original(*args, **kwargs)
        self.t_infer += time.perf_counter_ns() - t0
        return actions
    def _on_training_end(self):
        env_ms   = self.training_env.envs[0].env_time / 1e6 / self.steps
        infer_ms = self.t_infer / 1e6 / self.steps
        total_ms = self.t_tot  / 1e6 / self.steps
        over_ms  = total_ms - env_ms - infer_ms
        print({"steps/s": round(1000/total_ms,1),
               "env_ms":  round(env_ms,3),
               "infer_ms":round(infer_ms,3),
               "over_ms": round(over_ms,3)})