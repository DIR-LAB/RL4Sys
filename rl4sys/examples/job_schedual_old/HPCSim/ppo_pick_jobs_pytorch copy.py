# PyTorch implementation of PPO for HPC job scheduling. Converted from TensorFlow version.

import os
import time
import datetime
import csv
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym

from spinup.utils.logx import EpochLogger
from HPCSimPickJobs import (
    MAX_QUEUE_SIZE,
    JOB_FEATURES,
    JOB_SEQUENCE_SIZE,
    combined_shape,
    discount_cumsum,
    HPCEnv,
)
from torch.utils.tensorboard import SummaryWriter

#FILE_NAME= "SDSC-SP2-1998-4.2-cln"
#DATA_FILE= "./data/SDSC-SP2-1998-4.2-cln.swf"
#MODEL_FILE= "./data/SDSC-SP2-1998-4.2-cln.schd"

#FILE_NAME= "lublin_256"
#DATA_FILE= "./data/lublin_256.swf"
#MODEL_FILE= "./data/lublin_256.schd"

FILE_NAME= "SDSC-SP2-1998-4.2-cln"
DATA_FILE= "./data/SDSC-SP2-1998-4.2-cln.swf"
MODEL_FILE= "./data/SDSC-SP2-1998-4.2-cln.schd"
# -----------------------------------------------------------------------------
# Utility classes / helpers
# -----------------------------------------------------------------------------
class CustomCSVLogger:
    """Custom CSV logger that logs data in the same format as example.csv"""

    def __init__(self, log_dir: str = "./log") -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)

        self.csv_file = os.path.join(self.log_dir, f"{FILE_NAME}.csv")
        self.csv_writer = None
        self.csv_file_handle = None
        self._init_csv()

        self.writer = SummaryWriter(self.log_dir)

        # Episode tracking
        self.episode_count = 0
        self.iteration = 0
        self.client_id = "job_scheduling_pt"
        self.seed = 1

    def _init_csv(self) -> None:
        headers = [
            "timestamp",
            "iteration",
            "episode",
            "episode_steps",
            "reward",
            "done",
            "sjf_score",
            "f1_score",
            "episode_return",
            "client_id",
            "seed",
        ]
        self.csv_file_handle = open(self.csv_file, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file_handle)
        self.csv_writer.writerow(headers)

    def log_episode(
        self,
        episode_steps: int,
        reward: float,
        done: bool,
        sjf_score: float,
        f1_score: float,
        episode_return: float,
    ) -> None:
        timestamp = datetime.datetime.now().isoformat()
        row = [
            timestamp,
            self.iteration,
            self.episode_count,
            episode_steps,
            reward,
            done,
            sjf_score,
            f1_score,
            episode_return,
            self.client_id,
            self.seed,
        ]

        self.writer.add_scalar("reward", reward, self.episode_count)
        self.writer.add_scalar("sjf_score", sjf_score, self.episode_count)
        self.writer.add_scalar("f1_score", f1_score, self.episode_count)
        self.writer.add_scalar("episode_return", episode_return, self.episode_count)

        self.csv_writer.writerow(row)
        self.csv_file_handle.flush()
        if done:
            self.episode_count += 1

    def set_iteration(self, iteration: int) -> None:
        self.iteration = iteration

    def close(self) -> None:
        if self.csv_file_handle:
            self.csv_file_handle.close()


class PPOBuffer:
    """A simple numpy-based buffer for PPO with GAE"""

    def __init__(
        self,
        obs_dim: Tuple[int],
        act_dim: int,
        size: int,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        size = size * 100  # Same as original—large cap for trajectory length
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.int32)
        self.mask_buf = np.zeros(combined_shape(size, MAX_QUEUE_SIZE), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(
        self,
        obs: np.ndarray,
        act: int,
        mask: np.ndarray,
        rew: float,
        val: float,
        logp: float,
    ) -> None:
        assert self.ptr < self.max_size, "Buffer overflow"
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.mask_buf[self.ptr] = mask
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0.0) -> None:
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr < self.max_size, "Buffer overflow at get()"
        actual_size = self.ptr
        self.ptr, self.path_start_idx = 0, 0

        adv = self.adv_buf[:actual_size]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return (
            self.obs_buf[:actual_size],
            self.act_buf[:actual_size],
            self.mask_buf[:actual_size],
            adv,
            self.ret_buf[:actual_size],
            self.logp_buf[:actual_size],
        )


# -----------------------------------------------------------------------------
# Neural networks
# -----------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Combined actor-critic network."""

    def __init__(self, obs_dim: int, act_dim: int, attn: bool = False):
        super().__init__()
        self.attn = attn
        self.act_dim = act_dim

        # Shared per-job feature extractor (RL Kernel)
        self.fc1 = nn.Linear(JOB_FEATURES, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc_logits = nn.Linear(8, 1)

        # Value network operates on flattened observation
        self.v_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _rl_kernel(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.fc_logits(x).squeeze(-1)  # (batch, MAX_QUEUE_SIZE)
        return logits

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = obs.shape[0]
        obs_reshaped = obs.view(batch, MAX_QUEUE_SIZE, JOB_FEATURES)
        logits = self._rl_kernel(obs_reshaped)
        value = self.v_net(obs).squeeze(-1)
        return logits, value

    # ------------------------------------------------------------------
    # Act (with masking)
    # ------------------------------------------------------------------
    def act(self, obs: torch.Tensor, mask: torch.Tensor):
        logits, value = self.forward(obs)
        logits = logits + (mask - 1) * 1e6  # Mask invalid actions
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value, logits


# -----------------------------------------------------------------------------
# PPO Algorithm (Torch)
# -----------------------------------------------------------------------------

def ppo(
    workload_file: str,
    model_path: str,
    seed: int = 0,
    traj_per_epoch: int = 4000,
    epochs: int = 50,
    gamma: float = 0.99,
    clip_ratio: float = 0.2,
    pi_lr: float = 3e-4,
    vf_lr: float = 1e-3,
    train_pi_iters: int = 80,
    train_v_iters: int = 80,
    lam: float = 0.97,
    max_ep_len: int = 1000,
    target_kl: float = 0.01,
    logger_kwargs: dict = None,
    save_freq: int = 10,
    attn: bool = False,
    shuffle: bool = False,
    backfil: bool = False,
    skip: bool = False,
    score_type: int = 0,
    batch_job_slice: int = 0,
):

    logger_kwargs = logger_kwargs or {}
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    custom_logger = CustomCSVLogger()
    custom_logger.seed = seed

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Environment ----------------------------------------------------------------
    env = HPCEnv(
        shuffle=shuffle,
        backfil=backfil,
        skip=skip,
        job_score_type=score_type,
        batch_job_slice=batch_job_slice,
        build_sjf=False,
    )
    env.seed(seed)
    env.my_init(workload_file=workload_file, sched_file=model_path)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    ac = ActorCritic(obs_dim, act_dim, attn=attn)
    actor_optimizer = optim.Adam(
        [
            p for n, p in ac.named_parameters() if "v_net" not in n
        ],
        lr=pi_lr,
    )
    critic_optimizer = optim.Adam(ac.v_net.parameters(), lr=vf_lr)

    buf = PPOBuffer(obs_dim, 1, traj_per_epoch * JOB_SEQUENCE_SIZE, gamma, lam)

    # ------------------------------------------------------------------
    # Update function (uses closure over buffer / ac / optimizers)
    # ------------------------------------------------------------------
    def update():
        (
            obs_buf,
            act_buf,
            mask_buf,
            adv_buf,
            ret_buf,
            logp_old_buf,
        ) = buf.get()

        obs_t = torch.as_tensor(obs_buf, dtype=torch.float32)
        act_t = torch.as_tensor(act_buf, dtype=torch.long).squeeze(-1)
        mask_t = torch.as_tensor(mask_buf, dtype=torch.float32)
        adv_t = torch.as_tensor(adv_buf, dtype=torch.float32)
        ret_t = torch.as_tensor(ret_buf, dtype=torch.float32)
        logp_old_t = torch.as_tensor(logp_old_buf, dtype=torch.float32)

        # --------------------------------------------------------------
        # Train policy with multiple steps of gradient descent
        # --------------------------------------------------------------
        for i in range(train_pi_iters):
            logits, _ = ac.forward(obs_t)
            logits = logits + (mask_t - 1) * 1e6
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act_t)
            ratio = torch.exp(logp - logp_old_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_t
            pi_loss = -torch.mean(torch.min(surr1, surr2))

            actor_optimizer.zero_grad()
            pi_loss.backward()
            actor_optimizer.step()

            approx_kl = torch.mean(logp_old_t - logp).item()
            if approx_kl > 1.5 * target_kl:
                logger.log(f"Early stopping at step {i} due to reaching max kl.")
                break
        logger.store(StopIter=i)

        # --------------------------------------------------------------
        # Train value function
        # --------------------------------------------------------------
        for _ in range(train_v_iters):
            _, value = ac.forward(obs_t)
            v_loss = torch.mean((ret_t - value) ** 2)
            critic_optimizer.zero_grad()
            v_loss.backward()
            critic_optimizer.step()

        # Diagnostics --------------------------------------------------
        with torch.no_grad():
            logits, value = ac.forward(obs_t)
            logits = logits + (mask_t - 1) * 1e6
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act_t)
            ent = dist.entropy().mean().item()
            kl = torch.mean(logp_old_t - logp).item()
            ratio_diag = torch.exp(logp - logp_old_t)
            cf = (
                ((ratio_diag > (1 + clip_ratio)) | (ratio_diag < (1 - clip_ratio))).float().mean().item()
            )

        logger.store(
            LossPi=pi_loss.item(),
            LossV=v_loss.item(),
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=0.0,  # placeholder
            DeltaLossV=0.0,  # placeholder
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    o, r, d = env.reset()[0], 0.0, False  # env.reset returns ([o, co], ...)
    ep_ret, ep_len, show_ret, sjf, f1 = 0.0, 0, 0.0, 0.0, 0.0
    start_time = time.time()

    for epoch in range(epochs):
        custom_logger.set_iteration(epoch)
        t = 0
        action_list: List[int] = []
        while True:
            # Build mask (valid actions)
            lst = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                if all(o[i : i + JOB_FEATURES] == [0] + [1] * (JOB_FEATURES - 2) + [0]):
                    lst.append(0)
                elif all(o[i : i + JOB_FEATURES] == [1] * JOB_FEATURES):
                    lst.append(0)
                else:
                    lst.append(1)
            mask_np = np.array(lst, dtype=np.float32)

            obs_torch = torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
            mask_torch = torch.as_tensor(mask_np, dtype=torch.float32).unsqueeze(0)

            print("model parameters: ", ac.state_dict())
            exit()
            with torch.no_grad():
                action_t, logp_t, v_t, _ = ac.act(obs_torch, mask_torch)
            a = int(action_t.item())
            v_t = v_t.item()
            logp_t = logp_t.item()

            buf.store(o, a, mask_np, r, v_t, logp_t)
            logger.store(VVals=v_t)
            action_list.append(a)

            o, r, d, r2, sjf_t, f1_t = env.step(a)
            ep_ret += r
            ep_len += 1
            show_ret += r2
            sjf += sjf_t
            f1 += f1_t

            if d:
                buf.finish_path(r)
                logger.store(EpRet=ep_ret, EpLen=ep_len, ShowRet=show_ret, SJF=sjf, F1=f1)

                custom_logger.log_episode(
                    episode_steps=ep_len,
                    reward=r,
                    done=d,
                    sjf_score=sjf,
                    f1_score=f1,
                    episode_return=ep_ret,
                )

                o, r, d = env.reset()[0], 0.0, False
                ep_ret, ep_len, show_ret, sjf, f1 = 0.0, 0, 0.0, 0.0, 0.0

                # Count completed trajectories and break when desired number collected
                t += 1
                if t >= traj_per_epoch:
                    break
        # ----------------- Update after collecting trajectories --------
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            model_save_path = os.path.join(logger.output_dir, f"ac_model_{epoch}.pt")
            torch.save(ac.state_dict(), model_save_path)
            logger.save_state({"env": env}, None)

        update()

        # ----------------- Log at end of epoch --------------------------
        
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", with_min_and_max=True)
        logger.log_tabular("VVals", with_min_and_max=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * traj_per_epoch * JOB_SEQUENCE_SIZE)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossV", average_only=True)
        logger.log_tabular("DeltaLossPi", average_only=True)
        logger.log_tabular("DeltaLossV", average_only=True)
        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("ClipFrac", average_only=True)
        logger.log_tabular("StopIter", average_only=True)
        logger.log_tabular("ShowRet", average_only=True)
        logger.log_tabular("SJF", average_only=True)
        logger.log_tabular("F1", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.log_tabular("Epoch", epoch)
        logger.dump_tabular()

    custom_logger.close()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--trajs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--exp_name", type=str, default="ppo_pt")
    parser.add_argument("--attn", type=int, default=0)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--backfil", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--score_type", type=int, default=0)
    parser.add_argument("--batch_job_slice", type=int, default=0)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    current_dir = os.getcwd()
    #workload_file = os.path.join(current_dir, args.workload)
    log_data_dir = os.path.join(current_dir, "./data/logs/")
    logger_kwargs = setup_logger_kwargs(args.exp_name, seed=args.seed, data_dir=log_data_dir)

    ppo(
        DATA_FILE,
        MODEL_FILE,
        gamma=args.gamma,
        seed=args.seed,
        traj_per_epoch=args.trajs,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        attn=bool(args.attn),
        shuffle=bool(args.shuffle),
        backfil=bool(args.backfil),
        skip=bool(args.skip),
        score_type=args.score_type,
        batch_job_slice=args.batch_job_slice,
    ) 