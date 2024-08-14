import os
import json

class ConfigLoader:
    def __init__(self, config_path=None, algorithm=None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 'config.json')
        self.config = self.load_config()

        self.algorithm_params = self.get_algorithm_params(algorithm) if algorithm is not None else None
        self.train_server = self.get_train_server()
        self.traj_server = self.get_traj_server()
        self.tb_params = self.get_tensorboard_params()
        self.load_model_path = self.get_load_model_path()
        self.save_model_path = self.get_save_model_path()
        self.max_traj_length = self.get_max_traj_length()
    
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, KeyError):
            print(f"Failed to load configuration from {self.config_path}, loading defaults.")
            return {}

    def get_algorithm_params(self, algo: str):
        available_algorithms = ['DQN', 'PPO']
        if algo is None or algo not in available_algorithms:
            return None
        try:
            algorithm_params = self.config['algorithms'][algo]
        except FileNotFoundError:
            print(f"[ConfigLoader] Failed to load algorithm hyperparameters, loading defaults.")
            if algo == 'C51':
                algorithm_params = {
                    "batch_size": 32,
                    "seed": 0,
                    "traj_per_epoch": 3,
                    "atoms": 51,
                    "v_min": -10.0,
                    "v_max": 10.0,
                    "gamma": 0.95,
                    "epsilon": 1.0,
                    "epsilon_min": 0.01,
                    "epsilon_decay": 5e-4,
                    "q_lr": 1e-3,
                    "train_q_iters": 80
                }
            elif algo == 'DQN':
                algorithm_params = {
                    "batch_size": 32,
                    "seed": 0,
                    "traj_per_epoch": 3,
                    "gamma": 0.95,
                    "epsilon": 1.0,
                    "epsilon_min": 0.01,
                    "epsilon_decay": 5e-4,
                    "train_update_freq": 4,
                    "q_lr": 1e-3,
                    "train_q_iters": 80
                }
            elif algo == 'PPO':
                algorithm_params = {
                    "seed": 0,
                    "traj_per_epoch": 3,
                    "clip_ratio": 0.2,
                    "gamma": 0.99,
                    "lam": 0.97,
                    "pi_lr": 3e-4,
                    "vf_lr": 1e-3,
                    "train_pi_iters": 80,
                    "train_v_iters": 80,
                    "target_kl": 0.01,
                }
            else:
                algorithm_params = None
        return algorithm_params

    def get_train_server(self):
        try:
            train_server = self.config['server']['training_server']
        except KeyError:
            print("[ConfigLoader] Failed to load training server configuration, loading defaults.")
            train_server = {
                'prefix': 'tcp://',
                'host': '*',
                'port': ":5556"
            }
        return train_server

    def get_traj_server(self):
        try:
            traj_server = self.config['server']['trajectory_server']
        except KeyError:
            print("[ConfigLoader] Failed to load trajectory server configuration, loading defaults.")
            traj_server = {
                'prefix': 'tcp://',
                'host': 'localhost',
                'port': ":5555"
            }
        return traj_server
    
    def get_tensorboard_params(self):
        top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../RL4Sys'))
        try:
            tb_params = self.config['tensorboard']['tensorboard_writer']
        except KeyError:
            print("[ConfigLoader] Failed to load tensorboard parameters, loading defaults.")
            tb_params = {
                'tb_log_dir': 'utils/tb_runs',
                'data_log_dir': 'data',
                'scalar_tags': 'AverageEpRet;StdEpRet',
                'max_count_per_scalar': 100,
                'global_step_tag': 'Epoch'
            }
        
        tb_params['tb_log_dir'] = os.path.join(top_dir, tb_params['tb_log_dir'])
        tb_params['data_log_dir'] = os.path.join(top_dir, tb_params['data_log_dir'])
        return tb_params
    
    def get_load_model_path(self):
        try:
            load_model_path = os.path.join(os.path.dirname(__file__), self.config['model_paths']['load_model'])
        except KeyError:
            print("[ConfigLoader] Failed to load model path, loading defaults.")
            load_model_path = os.path.join(os.path.dirname(__file__), 'models/model.pth')
        return load_model_path
    
    def get_save_model_path(self):
        try:
            save_model_path = os.path.join(os.path.dirname(__file__), self.config['model_paths']['save_model'])
        except KeyError:
            print("[ConfigLoader] Failed to load save model path, loading defaults.")
            save_model_path = os.path.join(os.path.dirname(__file__), 'models/model.pth')
        return save_model_path

    def get_max_traj_length(self):
        try:
            max_traj_length = self.config['max_traj_length']
        except KeyError:
            print("[ConfigLoader] Failed to load max trajectory length, loading defaults.")
            max_traj_length = 1000
        return max_traj_length
    
