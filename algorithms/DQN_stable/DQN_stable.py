from _common._algorithms.BaseAlgorithm import AlgorithmAbstract
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch

import sys
import os
import zmq
import gymnasium as gym
import gymnasium.spaces as spaces

# We import stable-baselines3
from stable_baselines3 import DQN as SB3_DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import get_device

# from .kernel import DeepQNetwork    # OLD: We'll replace with stable-baselines
# from .replay_buffer import ReplayBuffer  # OLD: We'll replace with stable-baselines

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import EpochLogger, setup_logger_kwargs
from trajectory import RL4SysTrajectory

from conf_loader import ConfigLoader

config_loader = ConfigLoader(algorithm='DQN')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path


class DummyEnv(gym.Env):
    """
    A dummy environment to satisfy stable-baselines3 DQN’s requirement of an environment.
    We do not actually use environment rollouts; we feed transitions manually.

    Observations:
        Shape: (obs_dim, )

    Actions:
        Discrete with n_actions
    """
    def __init__(self, obs_dim=1, n_actions=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        
        # Here we define the observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

    def reset(self,seed=None):
        """
        Return a dummy observation of zeros.
        """
        info = {}
        return np.zeros(self.obs_dim, dtype=np.float32), info

    def step(self, action):
        """
        Return:
          - next observation
          - reward
          - done
          - info (dict)
        For a dummy environment, we do minimal logic.
        """
        next_obs = np.zeros(self.obs_dim, dtype=np.float32)
        reward = 0.0
        done = True
        truncated = False
        info = {}
        return next_obs, reward, done, truncated, info
    
class DQN_stable(AlgorithmAbstract):
    """
    This class now wraps stable-baselines3's DQN internally, but keeps
    the same interface your other scripts rely on.
    """

    def __init__(self, env_dir: str,
                 kernel_size: int,
                 kernel_dim: int,
                 buf_size: int,
                 act_dim: int = 1,
                 batch_size: int = hyperparams['batch_size'],
                 seed: int = hyperparams['seed'],
                 traj_per_epoch: int = hyperparams['traj_per_epoch'],
                 gamma: float = hyperparams['gamma'],
                 epsilon: float = hyperparams['epsilon'],
                 epsilon_min: float = hyperparams['epsilon_min'],
                 epsilon_decay: float = hyperparams['epsilon_decay'],
                 train_update_freq: float = hyperparams['train_update_freq'],
                 q_lr: float = hyperparams['q_lr'],
                 train_q_iters: int = hyperparams['train_q_iters']):

        super().__init__()
        # Set seeds
        seed += 10000 * os.getpid()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Save input parameters
        self._kernel_size = kernel_size
        self._kernel_dim = kernel_dim
        self._buf_size = buf_size
        self._act_dim = act_dim
        self._batch_size = batch_size

        # Hyperparameters
        self._traj_per_epoch = traj_per_epoch
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._train_update_freq = train_update_freq
        self._train_q_iters = train_q_iters

        # Create stable-baselines3 DQN instance
        # We pass a dummy environment, since we'll manually add transitions
        # to the replay buffer. We also set the buffer_size to buf_size.
        dummy_env = DummyVecEnv([lambda: DummyEnv(obs_dim=kernel_size * kernel_dim, n_actions=act_dim)])
        # dummy_env = gym.make('LunarLander-v3')
        self.model = SB3_DQN(
            "MlpPolicy",
            dummy_env,
            verbose=0,
            buffer_size=self._buf_size,
            batch_size=self._batch_size,
            gamma=self._gamma,
            learning_rate=q_lr,
            # We'll let stable-baselines handle epsilon. For a manual schedule, see doc:
            #   https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#using-optimizer-schedulers
            # Or we can override if we want a simpler approach. We'll just rely on stable-baselines' default exploration.
            seed=seed,
            device=get_device("cpu")  # or "cpu"/"cuda"
        )

        # stable-baselines3 automatically creates a replay buffer accessible via self.model.replay_buffer
        # if you want to manipulate it directly, you can reference it here:
        # self.model.replay_buffer = ...

        # set up logger
        log_data_dir = os.path.join(env_dir, './logs/')
        logger_kwargs = setup_logger_kwargs("rl4sys-dqn-info", seed=seed, data_dir=log_data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.logger.setup_pytorch_saver(self.model)  # We can save the model

        self.traj = 0
        self.epoch = 0

    
    def save(self, filename: str) -> None:
        """Save model as file (.pth extension)."""
        #new_path = os.path.join(save_model_path)
        # Using stable-baselines3's built-in saving (will be save into .zip format by default)

        #torch.save(self.model.policy.state_dict(), new_path)
        self.model.save('examples/maze-game/model.zip')


    def receive_trajectory(self, trajectory: RL4SysTrajectory) -> bool:
        """
        Receive a trajectory from a remote environment, store it in the SB3 replay buffer.
        Then, if we have enough trajectories, possibly call train_model().
        Returns:
            True if an epoch was triggered and an updated model should be sent.
        """
        self.traj += 1
        ep_ret, ep_len = 0, 0

        for r4a in trajectory.actions:
            ep_ret += r4a.rew
            ep_len += 1

            # stable-baselines3 replay buffer expects: obs, next_obs, action, reward, done, infos
            # Make sure obs & next_obs are the correct shapes
            obs = r4a.obs  # shape [kernel_dim*kernel_size] if that is how you store it
            next_obs = r4a.next_obs if hasattr(r4a, "next_obs") else obs  # fallback if next_obs not provided
            action = r4a.act
            reward = r4a.rew
            done = r4a.done

            # Convert obs, next_obs, etc. to np.array
            obs = np.array(obs, dtype=np.float32).reshape(-1)
            next_obs = np.array(next_obs, dtype=np.float32).reshape(-1)

            # Add to SB3’s replay buffer
            if not r4a.done:
                self.model.replay_buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                infos=[{}]
                )
                self.logger.store(QVals=r4a.data[0], Epsilon=r4a.data[0])
            else:
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)

        # We mimic your old approach: if enough trajectories have arrived, train the model
        # and return True so the training_server can push the new weights back to the agent.
        if self.traj > 0 and self.traj % self._traj_per_epoch == 0:
            if self.traj % self._train_update_freq == 0:
                self.epoch += 1
                # stop collecting staled traj during model training
                self.client_stop_collect_traj('stop')

                self.train_model()
                # Log stats
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                self.log_epoch()
                return True

        return False

    def client_stop_collect_traj(self, msg):
        """
        'stop' means stop collecting stale traj during model training
        """
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        socket.connect("tcp://127.0.0.1:5554")  # TODO fix after
        print("Server told Client stop collecting on port 5554")
        socket.send_string(msg)
        socket.close()
        context.term()

    def train_model(self) -> None:
        """
        Train model on data from stable-baselines3’s internal replay_buffer.
        We do a hacky approach: we call .train() for some gradient steps.
        This is not a publicly documented method; use carefully.
        Alternatively, you can do: model.learn(total_timesteps=..., reset_num_timesteps=False),
        but that also expects environment steps.
        """
        # "gradient_steps" is how many gradient updates we want to do
        # We are reusing stable-baselines3’s internal mechanism for training
        # each call to .train() will sample from the replay buffer.
        gradient_steps = self._train_q_iters

        # We must simulate environment steps to trick SB3 into doing gradient updates
        # or we call .learn with total_timesteps = gradient_steps, but that confuses the environment logic.
        # Instead, we force the training loop manually:
        self.model.env.reset()
        # Let’s do gradient_steps updates in a loop:
        for _ in range(gradient_steps):
            # This function is internal, but it is how SB3 DQN does a training iteration
            self.model.learn(
            total_timesteps=10,
            reset_num_timesteps=False,
            progress_bar=False
        )

        # For logging, we store a simple placeholder
        self.logger.store(StopIter=gradient_steps)
        self.logger.store(QTargets=0.0, LossQ=0.0, DeltaLossQ=0.0)

    def log_epoch(self) -> None:
        """Log the information collected in logger over the course of the last epoch"""
        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('Epsilon', with_min_and_max=True)
        self.logger.log_tabular('QVals', average_only=True)
        self.logger.log_tabular('QTargets', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.log_tabular('DeltaLossQ', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)
        self.logger.dump_tabular()

    def compute_loss_q(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """
        We keep this function for backward compatibility with your code,
        but stable-baselines3 does not expose direct Q-loss computations
        in an open interface. We return dummy placeholders.
        """
        # If some part of your code calls this, you might fill this with
        # advanced logic that inspects self.model.replay_buffer or self.model.policy.
        dummy_loss = torch.tensor(0.0)
        dummy_q_target = np.array([0.0])
        return dummy_loss, dummy_q_target
