from _common._algorithms.BaseAlgorithm import AlgorithmAbstract

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .kernel import DeepQNetwork
from .replay_buffer import ReplayBuffer

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.logger import EpochLogger, setup_logger_kwargs
from trajectory import RL4SysTrajectory

from conf_loader import ConfigLoader

import zmq

"""
Import and load RL4Sys/config.json DQN Agent configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
config_loader = ConfigLoader(algorithm='DQN')
hyperparams = config_loader.algorithm_params
save_model_path = config_loader.save_model_path


"""
DQN Agent with hyperparameters
"""


class DQN(AlgorithmAbstract):
    """
            Args:
                kernel_size: number of observations
                kernel_dim: number of features
                buf_size: size of replay buffer
                act_dim: number of actions (output dimension(s))
                batch_size: batch size of replay buffer
                seed: seed for random number generator
                traj_per_epoch: number of trajectories to be retrieved prior to training
                gamma: Q target discount factor
                epsilon: initial value for epsilon; exploration rate that is decayed over time
                epsilon_min: minimum value for epsilon
                epsilon_decay: decay rate for epsilon
                train_update_freq: frequency of training model
                q_lr: learning rate for Q network, passed to Adam optimizer
                train_q_iters:
    """
    def __init__(self, env_dir: str, kernel_size: int, kernel_dim: int, act_dim: int, buf_size: int,
                 batch_size: int = hyperparams['batch_size'], seed: int = hyperparams['seed'],
                 traj_per_epoch: int = hyperparams['traj_per_epoch'], gamma: float = hyperparams['gamma'],
                 epsilon: float = hyperparams['epsilon'], epsilon_min: float = hyperparams['epsilon_min'],
                 epsilon_decay: float = hyperparams['epsilon_decay'],
                 train_update_freq: float = hyperparams['train_update_freq'], 
                 q_lr: float = hyperparams['q_lr'],
                 train_q_iters: int = hyperparams['train_q_iters'],
                 target_net_update_frequency:int = hyperparams['target_net_update_frequency']):

        super().__init__()
        seed += 10000 * os.getpid()
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Input parameters
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
        self._target_net_update_frequency = target_net_update_frequency # update target net

        self._replay_buffer = ReplayBuffer(kernel_size * kernel_dim, kernel_size, buf_size, gamma=gamma, epsilon=epsilon)
        self._model = DeepQNetwork(kernel_size, kernel_dim, act_dim, self._epsilon, self._epsilon_min, self._epsilon_decay)
        self.q_target = DeepQNetwork(kernel_size, kernel_dim, act_dim, self._epsilon, self._epsilon_min, self._epsilon_decay)
        self.q_target.load_state_dict(self._model.state_dict())
        
        self._q_optimizer = Adam(self._model.parameters(), lr=q_lr)

        # set up logger
        log_data_dir = os.path.join(env_dir, './logs/')
        logger_kwargs = setup_logger_kwargs(
            "rl4sys-dqn-info", seed=seed, data_dir=log_data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.logger.setup_pytorch_saver(self._model)

        self.traj = 0
        self.epoch = 0

        # ADDED: for q target network
        self.target_update_freq = 500  # how often to sync weights
        self.total_steps = 0

    def save(self, filename: str) -> None:
        """Save model as file.

        Uses .pth file extension.

        Args:
            filename: name to save file as

        """
        new_path = os.path.join(save_model_path, filename +
                                ('.pth' if not filename.__contains__('.pth') else ''))
        torch.save(self._model, new_path)

    def receive_trajectory(self, trajectory: RL4SysTrajectory) -> bool:
        """Process a trajectory received by training_server.

        If an epoch is triggered, calls train_model().

        Args:
            trajectory: holds agent experiences since last trajectory
        Returns:
            True if an epoch was triggered and an updated model should be sent.

        """
        self.traj += 1
        ep_ret, ep_len = 0, 0

        for r4a in trajectory.actions:
            # Process each RL4SysAction in the trajectory
            ep_ret += r4a.rew
            ep_len += 1
            if not r4a.done:
                self._replay_buffer.store(r4a.obs, r4a.next_obs, r4a.act, r4a.mask, r4a.rew, r4a.data['q_val'])
                self.logger.store(QVals=r4a.data['q_val'], Epsilon=r4a.data['epsilon'])
            else:
                self._replay_buffer.finish_path(r4a.rew)
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)

        # get enough trajectories for training the model
        if self.traj > 0 and self.traj % self._traj_per_epoch == 0:
            self.epoch += 1
            # TODO send client a signal to tell client ingore all trajectory collected. Resume after client model update
            self.client_stop_collect_traj('stop')

            self.train_model()
            self.log_epoch()
            return True

        return False

    def client_stop_collect_traj(self, msg):
        """
        msg = "stop" means stop collecting staled traj during model training
        """
        context = zmq.Context()
        socket = context.socket(zmq.PUSH)  # REP socket for replies
        socket.connect("tcp://127.0.0.1:5554") # TODO fix after
        print("Server told Client stop collecting on port 5554")
        socket.send_string(msg)

        socket.close()
        context.term()

    def train_model(self) -> None:
        """Train model on data from DQN replay_buffer.
        """
        data, batch = self._replay_buffer.get(self._batch_size)

        q_l_old = self.compute_loss_q(data)[0]
        q_l_old = q_l_old.item()

        # Train Q network for n iterations of gradient descent
        for i in range(self._train_q_iters):
            self._q_optimizer.zero_grad()
            loss_q, q_target = self.compute_loss_q(data)
            loss_q.backward()
            self._q_optimizer.step()

            self.total_steps += 1
            if self.total_steps % self.target_update_freq == 0:
                self.q_target.load_state_dict(self._model.state_dict())

        self.logger.store(StopIter=i)

        self.logger.store(QTargets=q_target, LossQ=loss_q.item(), DeltaLossQ=abs(loss_q.item() - q_l_old))

        

    def log_epoch(self) -> None:
        """Log the information collected in logger over the course of the last epoch
        """
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
        """Compute loss for Q function.

        Args:
            data: dictionary containing batched data from replay buffer
        Returns:
            Loss for Q function, Q target for logging

        """
        mask = data['mask']
        obs, act, rew, next_obs = data['obs'], data['act'], data['rew'], data['next_obs']

        # Q loss
        q_val = self._model.forward(obs, mask)
        q_taken = q_val.gather(1, act.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            q_next_values = self.q_target(next_obs, mask) # ADDED
            q_next_max = q_next_values.max(dim=1)[0]
            q_targ = rew + self._gamma * q_next_max

        # Mean Square Error (MSE) loss
        loss_q = ((q_taken - q_targ)**2).mean()

        return loss_q, q_targ.detach().numpy()
