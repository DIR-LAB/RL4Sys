import os
import json
import sys # Added for sys.exit


class ConfigLoader:
    def __init__(self, config_path=None):
        # Load the main configuration file
        self.config_path = config_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.json')
        self.config = self.load_config()

        # Determine and validate the default algorithm
        try:
            self.default_algorithm = self.config['default_algorithm']
        except KeyError:
            print(f"Error: 'default_algorithm' key not found in configuration file: {self.config_path}")
            sys.exit(1)

        if self.default_algorithm not in self.config.get('algorithms', {}):
            print(f"Error: Default algorithm '{self.default_algorithm}' specified in config not found in the 'algorithms' section.")
            sys.exit(1)
        
        # Load parameters for the validated default algorithm
        self.algorithm_params = self.config['algorithms'][self.default_algorithm]
        print(f"[ConfigLoader] Loaded parameters for default algorithm: {self.default_algorithm}")

        # Get algorithm type and limits before removing them
        self.algorithm_type = self.algorithm_params.get('type', 'onpolicy')  # Default to onpolicy if not specified
        self.max_traj_length = self.algorithm_params.get('max_traj_length', 1000)  # Default to 1000 if not specified
        self.act_limit = self.algorithm_params.get('act_limit', 1.0)  # Default to 1.0 if not specified

        # Remove certain parameters that shouldn't be passed to algorithm initialization
        exclude_params = ['type', 'max_traj_length', 'act_limit']
        for param in exclude_params:
            self.algorithm_params.pop(param, None)

        # Load other general configurations
        self.train_server_address = self.get_train_server_address()
        self.tb_params = self.get_tensorboard_params()
        self.load_model_path = self.get_load_model_path()
        self.save_model_path = self.get_save_model_path()
    
    def get_algorithm_type(self):
        return self.algorithm_type
    
    def get_max_traj_length(self):
        return self.max_traj_length
    
    def get_act_limit(self):
        return self.act_limit
    
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Could not parse configuration file {self.config_path}: {e}")
            sys.exit(1)
        except Exception as e: # Catch other potential errors during file reading
            print(f"Error: An unexpected error occurred while loading configuration from {self.config_path}: {e}")
            sys.exit(1)

    def get_train_server_address(self):
        try:
            train_server_address = self.config['server']['training_server']
        except KeyError:
            print("[ConfigLoader] Warning: Training server address not found in config, using default: localhost:50051")
            train_server_address = "localhost:50051"
        return train_server_address
    
    def get_tensorboard_params(self):
        top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../RL4Sys'))
        try:
            tb_params = self.config['tensorboard']['tensorboard_writer']
        except KeyError:
            print("[ConfigLoader] Warning: Tensorboard parameters not found in config, using defaults.")
            tb_params = {
                'scalar_tags': 'AverageEpRet;StdEpRet',
                'max_count_per_scalar': 100,
                'global_step_tag': 'Epoch'
            }

        return tb_params
    
    def get_load_model_path(self):
        # Construct path relative to project root (where config.json is)
        project_root = os.path.dirname(self.config_path)
        try:
            relative_path = self.config['model_paths']['load_model']
            load_model_path = os.path.abspath(os.path.join(project_root, relative_path))
        except KeyError:
            print("[ConfigLoader] Warning: Load model path not found in config, using default: models/model.pth relative to project root.")
            load_model_path = os.path.abspath(os.path.join(project_root, 'models/model.pth'))
        return load_model_path
    
    def get_save_model_path(self):
        # Construct path relative to project root (where config.json is)
        project_root = os.path.dirname(self.config_path)
        try:
            relative_path = self.config['model_paths']['save_model']
            save_model_path = os.path.abspath(os.path.join(project_root, relative_path))
        except KeyError:
            print("[ConfigLoader] Warning: Save model path not found in config, using default: models/model.pth relative to project root.")
            save_model_path = os.path.abspath(os.path.join(project_root, 'models/model.pth'))
        return save_model_path
    
