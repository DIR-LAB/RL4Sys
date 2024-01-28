import os
import pickle
import threading
import numpy as np

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.distributions.categorical import Categorical
from action import RL4SysAction
from models.kernel_ac import RLActorCritic

from reply_buffer import ReplayBuffer
import zmq
from utils.logger import EpochLogger, setup_logger_kwargs  # Assuming Trajectory class is defined

class PPOTrainingServer:
    def __init__(self, flatten_obs_dim, kernel_size, kernel_dim,
                 obs_dim, act_dim, mask_dim, size, seed=0,
                 traj_per_epoch=3, epochs=100,
                 clip_ratio=0.2, pi_lr=3e-4,
                 vf_lr=1e-3, train_pi_iters=80, 
                 train_v_iters=80, lam=0.97, max_ep_len=1000,
                 target_kl=0.01):
        
        current_dir = os.getcwd()
        log_data_dir = os.path.join(current_dir, './logs/')

        logger_kwargs = setup_logger_kwargs(
            "rl4sys-ppo-scheduler", seed=seed, data_dir=log_data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        seed += 10000 * os.getpid()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len
        self.target_kl = target_kl
        self.traj_per_epoch = traj_per_epoch
        self.epochs = epochs
        
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, mask_dim, size)

        # create the model ready for training.
        self.model = RLActorCritic(flatten_obs_dim, kernel_size, kernel_dim)

        self.pi_optimizer = Adam(self.model.pi.parameters(), lr=self.pi_lr)
        self.vf_optimizer = Adam(self.model.v.parameters(), lr=self.vf_lr)

        self.logger.setup_pytorch_saver(self.model)
        
        # send the initial model in a different thread
        print("[TraingServer] Finish Initilizating, Sending the model...")
        self.initial_send_thread = threading.Thread(target=self.send_model)
        self.initial_send_thread.start()

        self.loop_thread = threading.Thread(target=self.start_loop)
        self.loop_thread.start()

    def joins(self):
        self.initial_send_thread.join()
        self.loop_thread.join()

    # server sends the updated model to client 
    def send_model(self):
        print("[TraingServer - send_model] Send a model to RL4SysAgent")
        
        torch.save(self.model, 'model.pth')

        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        # replace with actual address and port
        socket.connect("tcp://localhost:5556")

        with open('model.pth', 'rb') as f:
            b = f.read()
            socket.send(b)
        
        socket.close()
        context.term()


    # Set up function for computing PPO policy loss
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old, mask = data['obs'], data['act'], data['adv'], data['logp'], data['mask']

        # Policy loss
        pi, logp = self.model.pi(obs, mask, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self, data):
        obs, ret, mask = data['obs'], data['ret'], data['mask']
        return ((self.model.v(obs, mask) - ret)**2).mean()

    def start_loop(self):
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind("tcp://*:5555")

        traj = 0
        epoch = 0

        while True:
            print("[training_server.py - start_loop - blocking for new trajectory]")
            trajectory_data = socket.recv()
            traj += 1
            trajectory = pickle.loads(trajectory_data)
            ep_ret, ep_len = 0, 0
            
            for r4a in trajectory.actions:
                # Process each RL4SysAction in the trajectory
                ep_ret += r4a.rew
                ep_len += 1
                if not r4a.done:
                    self.replay_buffer.store(r4a.obs, r4a.act, r4a.mask, r4a.rew, r4a.val, r4a.logp)
                    self.logger.store(VVals=r4a.val)
                else:
                    self.replay_buffer.finish_path(r4a.rew)
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
            
            print("[training_server.py - start_loop - received traj: ]", traj)

            # get enough trajectories for training the model
            if traj >= self.traj_per_epoch:
                epoch += 1              
                self.train_model()
                # Log info about epoch
                self.logger.log_tabular('Epoch', epoch)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', with_min_and_max=True)
                self.logger.log_tabular('VVals', with_min_and_max=True)
                self.logger.log_tabular('LossPi', average_only=True)
                self.logger.log_tabular('LossV', average_only=True)
                self.logger.log_tabular('DeltaLossPi', average_only=True)
                self.logger.log_tabular('DeltaLossV', average_only=True)
                self.logger.log_tabular('Entropy', average_only=True)
                self.logger.log_tabular('KL', average_only=True)
                self.logger.log_tabular('ClipFrac', average_only=True)
                self.logger.log_tabular('StopIter', average_only=True)
                self.logger.dump_tabular()

                # after finish training, we need to reset things and sync the model
                self.send_model()

        socket.close()
        context.term()

    def train_model(self):
        data = self.replay_buffer.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = (pi_info['kl'])
            if kl > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            self.pi_optimizer.step()
        
        self.logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))
