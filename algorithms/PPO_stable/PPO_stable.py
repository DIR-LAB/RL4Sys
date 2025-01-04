from _common._algorithms.BaseAlgorithm import AlgorithmAbstract
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import zmq
import gymnasium as gym
import gymnasium.spaces as spaces

# Instead of DQN, import PPO from stable-baselines3
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import get_device

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import EpochLogger, setup_logger_kwargs
from trajectory import RL4SysTrajectory

from conf_loader import ConfigLoader

config_loader = ConfigLoader(algorithm='PPO')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path


class DummyEnv(gym.Env):
    """
    A dummy environment to satisfy stable-baselines3 PPO’s requirement of an environment.
    We do not actually use environment rollouts meaningfully;
    we feed transitions manually (for logging),
    but PPO typically trains on-policy from real rollouts.
    """
    def __init__(self, obs_dim=1, n_actions=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

    def reset(self, seed=None):
        """
        Return a dummy observation (zeros), and an empty info dict.
        """
        super().reset(seed=seed)
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        """
        Return:
          - next observation
          - reward
          - done
          - truncated
          - info
        In a real env, done and truncated logic might differ.
        """
        next_obs = np.zeros(self.obs_dim, dtype=np.float32)
        reward = 0.0
        done = True
        truncated = False
        info = {}
        return next_obs, reward, done, truncated, info


class PPO_stable(AlgorithmAbstract):
    """
    This class now wraps stable-baselines3's PPO using the new hyperparams:
        - batch_size
        - seed
        - traj_per_epoch
        - clip_ratio (-> clip_range)
        - gamma
        - lam (-> gae_lambda)
        - pi_lr
        - vf_lr
        - train_pi_iters (-> n_epochs)
        - train_v_iters (not separately used, SB3 updates policy+value together)
        - target_kl
    """

    def __init__(self, env_dir: str,
                 kernel_size: int,
                 kernel_dim: int,
                 buf_size: int,     # Not used by on-policy PPO, but kept for compatibility
                 act_dim: int = 1,
                 batch_size: int = hyperparams['batch_size'],
                 seed: int = hyperparams['seed'],
                 traj_per_epoch: int = hyperparams['traj_per_epoch'],
                 clip_ratio: float = hyperparams['clip_ratio'],
                 gamma: float = hyperparams['gamma'],
                 lam: float = hyperparams['lam'],
                 pi_lr: float = hyperparams['pi_lr'],
                 vf_lr: float = hyperparams['vf_lr'],
                 train_pi_iters: int = hyperparams['train_pi_iters'],
                 train_v_iters: int = hyperparams['train_v_iters'],
                 target_kl: float = hyperparams['target_kl']):

        super().__init__()
        # Adjust seed for multiple processes
        seed += 10000 * os.getpid()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Save input parameters
        self._kernel_size = kernel_size
        self._kernel_dim = kernel_dim
        self._buf_size = buf_size  # Not typically used by PPO
        self._act_dim = act_dim
        self._batch_size = batch_size

        # Hyperparameters
        self._traj_per_epoch = traj_per_epoch
        self._clip_ratio = clip_ratio
        self._gamma = gamma
        self._lam = lam
        self._pi_lr = pi_lr
        self._vf_lr = vf_lr
        self._train_pi_iters = train_pi_iters
        self._train_v_iters = train_v_iters
        self._target_kl = target_kl

        # Create stable-baselines3 PPO instance
        obs_dim = kernel_size * kernel_dim
        dummy_env = DummyVecEnv([lambda: DummyEnv(obs_dim=obs_dim, n_actions=act_dim)])

        # SB3 PPO does not have separate pi_lr/vf_lr by default.
        # We'll use pi_lr as the single 'learning_rate' for both networks.
        # clip_range=clip_ratio, gae_lambda=lam, target_kl=target_kl, n_epochs=train_pi_iters
        self.model = SB3_PPO(
            "MlpPolicy",
            dummy_env,
            verbose=0,
            gamma=self._gamma,
            gae_lambda=self._lam,
            clip_range=self._clip_ratio,
            learning_rate=self._pi_lr,
            batch_size=self._batch_size,
            n_epochs=self._train_pi_iters,
            target_kl=self._target_kl,
            seed=seed,
            device=get_device("cpu")
        )

        # set up logger
        log_data_dir = os.path.join(env_dir, './logs/')
        logger_kwargs = setup_logger_kwargs("rl4sys-ppo-info", seed=seed, data_dir=log_data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.logger.setup_pytorch_saver(self.model)

        self.traj = 0
        self.epoch = 0

    def save(self, filename: str) -> None:
        """Save model as file (SB3 .zip)."""
        new_path = os.path.join(save_model_path, filename + ".zip")
        self.model.save(new_path)

    def receive_trajectory(self, trajectory: RL4SysTrajectory) -> bool:
        """
        In pure PPO, transitions are usually gathered from environment rollouts (model.learn).
        We'll collect stats here for code compatibility.
        Returns True if an epoch triggers an update to the model.
        """
        self.traj += 1
        ep_ret, ep_len = 0, 0

        for r4a in trajectory.actions:
            ep_ret += r4a.rew
            ep_len += 1

            # Log if the old code has Q-values or anything else
            if not r4a.done:
                # e.g. might store QVal or similar
                # Just an example if your old code had "q_val"
                self.logger.store(QVals=r4a.data[0])
            else: 
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)

        if self.traj > 0 and self.traj % self._traj_per_epoch == 0:
            self.epoch += 1
            # Potentially train the model once enough trajectories have arrived
            # or your old logic might have a separate condition, e.g. "train_update_freq".
            # self.client_stop_collect_traj('stop')
            self.train_model()
            self.logger.store(EpRet=ep_ret, EpLen=ep_len)
            self.log_epoch()
            return True

        return False

    def client_stop_collect_traj(self, msg):
        """
        'stop' means stop collecting stale transitions during model training
        """
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.connect("tcp://127.0.0.1:5554")  # adapt as needed
        print("Server told Client stop collecting on port 5554")
        socket.send_string(msg)
        socket.close()
        context.term()

    def train_model(self) -> None:
        """
        A hacky approach: we call .learn(...) for small timesteps in a loop.
        In typical PPO usage, you do one large call to model.learn(...).
        """
        # Combine train_pi_iters and train_v_iters if you want, 
        # but SB3 merges them in n_epochs.
        for _ in range(self._train_pi_iters):
            # Each call runs PPO for some environment timesteps
            self.model.learn(
                total_timesteps=10,  # or any small chunk you want
                reset_num_timesteps=False,
                progress_bar=False
            )

        self.logger.store(StopIter=self._train_pi_iters)
        self.logger.store(QTargets=0.0, LossQ=0.0, DeltaLossQ=0.0)

    def log_epoch(self) -> None:
        """Log the information collected in logger over the course of the last epoch."""
        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('QVals', average_only=True)
        self.logger.log_tabular('QTargets', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('DeltaLossQ', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)
        self.logger.dump_tabular()

    def compute_loss_q(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """
        PPO does not expose Q-value computations. Return dummy placeholders.
        """
        dummy_loss = torch.tensor(0.0)
        dummy_q_target = np.array([0.0])
        return dummy_loss, dummy_q_target
