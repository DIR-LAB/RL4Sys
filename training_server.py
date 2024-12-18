from _common._rl4sys.BaseTrainingServer import RL4SysTrainingServerAbstract

import argparse
import zmq
import threading
import pickle

import importlib
import inspect
from typing import Union
from typing import NoReturn as Never

from training_tensorboard import TensorboardWriter
from conf_loader import ConfigLoader
from time import time

ALGORITHMS_PATH = 'algorithms'

import os, json
"""Import and load RL4Sys/config.json server configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
config_loader = ConfigLoader()
train_server = config_loader.train_server
traj_server = config_loader.traj_server
save_model_path = config_loader.save_model_path


class TrainingServer(RL4SysTrainingServerAbstract):
    """Train a model for a remote agent.

    Sends one untrained model for agent initialization.
    Receives trajectories over the network.
    Trains model based on trajectories.
    Model is sent asynchronously back to agent whenever it is updated with new training.

    Can initalize algorithm object using python dict or command line arguments.

    Args:
        kernel_size (int): number of observations. e.g. MAX_QUEUE_SIZE
        kernel_dim (int): number of features. e.g. JOB_FEATURES
        hyperparams: hyperparameters specific to algorithm. Keys/flags correspond to algorithm class constructor.

    """
    def __init__(self, algorithm_name: str, kernel_size: int, kernel_dim: int, hyperparams: Union[dict | list[str]],
                 env_dir: str = os.getcwd(), tensorboard: bool = False):
        super().__init__(algorithm_name, obs_size=kernel_size, obs_dim=kernel_dim, hyperparams=hyperparams,
                         env_dir=env_dir)
        # get algorithm class
        algorithm_module: str = ALGORITHMS_PATH + ".{}".format(algorithm_name) + ".{}".format(algorithm_name)
        algorithm_module: importlib.ModuleType = importlib.import_module(algorithm_module)
        algorithm_class = getattr(algorithm_module, algorithm_name)
        
        # if hyperparams are a list of command-line arguments, parse into a hyperparam dict
        if isinstance(hyperparams, list):
            parser = argparse.ArgumentParser()

            # add each parameter of algorithm class
            parameters = inspect.signature(algorithm_class.__init__).parameters
            no_parse = ('env_dir', 'kernel_size', 'kernel_dim', 'self')
            for parameter in parameters.values():
                if parameter.name in no_parse:
                    continue # parameter has already been taken out
                flag = '--' + parameter.name
                type = parameter.annotation
                default = parameter.default
                if type is inspect.Parameter.empty:
                    type = None # if no type was hinted, cause a problem
                if default is inspect.Parameter.empty:
                    parser.add_argument(flag, type=type, required=True)
                else:
                    parser.add_argument(flag, type=type, default=default)

            args = parser.parse_args(hyperparams) # Raises error if any hyperparams are unrecognized in algorithm class
            
            hyperparams = vars(args)  # convert to dict

        # add TrainingServer args to algorithm parameters
        hyperparams['env_dir'] = env_dir
        hyperparams['kernel_size'] = kernel_size
        hyperparams['kernel_dim'] = kernel_dim

        # instantiate algorithm class
        self._algorithm = algorithm_class(**hyperparams)

        if tensorboard:
            self._tensorboard = TensorboardWriter(env_dir=env_dir, algorithm_name=algorithm_name)

        
        # add a trajectory buffer to asynchronizly store trajs and then dispatch to agent
        self.server_traj_buffer = [] # queue, FIFO

        # send the initial model in a different thread so we can start listener immediately
        print("[TrainingServer] Finish Initilizating, Sending the model...")
        self.initial_send_thread = threading.Thread(target=self.send_model)
        self.initial_send_thread.start()

        # start listener in a seperate thread
        self.loop_thread = threading.Thread(target=self.start_loop)
        self.loop_thread.start()



    # TODO ask why this exists
    def joins(self) -> None:
        """Wait until both of the following threads complete.

        * initial send thread
        * listener thread
        """
        self.initial_send_thread.join()
        self.loop_thread.join()

    def send_model(self) -> None:
        """Save the model and send it to the agent.
        """
        print("[TrainingServer - send_model] Send a model to RL4SysAgent")
        
        # TODO allow a change in name for saving model
        self._algorithm.save(save_model_path)

        context = zmq.Context()
        socket = context.socket(zmq.PUSH)
        address = f"{train_server['prefix']}{train_server['host']}{train_server['port']}"
        socket.connect(address)

        with open(save_model_path, 'rb') as f:
            b = f.read()
            socket.send(b)
        
        socket.close()
        context.term()

    def start_loop(self) -> Never:
        """Listens for a trajectories.

        After initialization, all server activity occurs on the thread running this loop.
        """
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        address = f"{traj_server['prefix']}{traj_server['host']}{traj_server['port']}"
        socket.bind(address)

        while True:
            print("[training_server.py - start_loop - blocking for new trajectory]")
            trajectory_data = socket.recv()
            trajectory = pickle.loads(trajectory_data)
            print("[training_server.py - start_loop - received traj #{}]".format(self._algorithm.traj))

            updated = self._algorithm.receive_trajectory(trajectory)
            if updated:
                self.send_model()
            
        socket.close()
        context.term()

            


if __name__ == "__main__":

    # Example:
    # python ./training_server.py PPO --size=12 --dim=5 --buf_size=(256 * 100) --gamma=.90
    parser = argparse.ArgumentParser(prog="RL4Sys Training Server",
                                     epilog="Pass algorithm-specific parameters according to class attribute names:" + "\n" +
                                                "  e.g. --buf_size=5 --gamma=.65",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('algorithm', type=str,
                        help="name of algorithm to use, as found in RL4Sys/algorithms")
    parser.add_argument('--kernel_size', '--size', type=int, required=True,
                        help="number of actions in each observation")
    parser.add_argument('--kernel_dim', '--dim', type=int, required=True,
                        help="number of features per action")
    args, extras = parser.parse_known_args()

    rl_training_server = TrainingServer(args.algorithm, args.kernel_size, args.kernel_dim, extras)