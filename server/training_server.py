import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import zmq
import threading
import pickle

import importlib
import inspect
from typing import Union
from typing import NoReturn as Never

from server.training_tensorboard import TensorboardWriter
from utils.conf_loader import ConfigLoader
from utils.util import deserialize_action

import time
import grpc
from concurrent import futures
from protocol import trajectory_pb2
from protocol import trajectory_pb2_grpc

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
hyperparam_server = config_loader.algorithm_params


class TrainingServer(trajectory_pb2_grpc.RL4SysRouteServicer):
    """Train a model for a remote agent.

    Sends one untrained model for agent initialization.
    Receives trajectories over the network.
    Trains model based on trajectories.
    Model is sent asynchronously back to agent whenever it is updated with new training.

    Can initalize algorithm object using python dict or command line arguments.
    """
    def __init__(self, algorithm_name: str, input_size: int, action_dim: int, hyperparams: Union[dict | list[str]],
                 env_dir: str = os.getcwd(), tensorboard: bool = False, seed = 0):
        super().__init__(algorithm_name, input_size==input_size, hyperparams=hyperparams,
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
            no_parse = ('env_dir', 'input_size', 'act_dim', 'self')
            for parameter in parameters.values():
                if parameter.name in no_parse:
                    continue # parameter has already been taken out
                flag = '--' + parameter.name
                type1 = parameter.annotation
                default = parameter.default
                if type1 is inspect.Parameter.empty:
                    type1 = None # if no type was hinted, cause a problem
                if default is inspect.Parameter.empty:
                    parser.add_argument(flag, type=type1, required=True)
                else:
                    parser.add_argument(flag, type=type1, default=default)

            args = parser.parse_args(hyperparams) # Raises error if any hyperparams are unrecognized in algorithm class
            
            hyperparams = vars(args)  # convert to dict

        # add TrainingServer args to algorithm parameters
        hyperparams['env_dir'] = env_dir
        hyperparams['input_size'] = input_size
        hyperparams['act_dim'] = action_dim

        # instantiate algorithm class
        self._algorithm = algorithm_class(**hyperparams)

        # for server components
        self.idle_timeout = 30 # TODO do it in config loader
        self.lock = threading.Lock()
        self.model_ready = 0
        self.trained_model_path = os.path.join(os.path.dirname(__file__), save_model_path, f"{algorithm_name}_model.pth")
        self.trained_model = None
        self.error_message = None

        if tensorboard:
            self._tensorboard = TensorboardWriter(env_dir=env_dir, algorithm_name=algorithm_name)

        # add a trajectory buffer to asynchronizly store trajs and then dispatch to agent
        self.server_traj_buffer = [] # queue, FIFO

        # send the initial model in a different thread so we can start listener immediately
        print("[TrainingServer] Finish Initilizating, Sending the model...")
        self.initial_send_thread = threading.Thread(target=self.send_model)
        self.initial_send_thread.daemon = True
        self.initial_send_thread.start()

        # start listener in a seperate thread
        self._loop_thread_stop_signal = threading.Event()
        self.loop_thread = threading.Thread(target=self.start_loop)
        self.loop_thread.daemon = True
        self.loop_thread.start()

    def save_model(self, model_data):
        """Save model to persistent storage."""
        os.makedirs(os.path.join(os.path.dirname(__file__), save_model_path), exist_ok=True)  # Ensure the directory exists
        with open(self.trained_model_path, "wb") as f:
            f.write(model_data)

    def load_model(self):
        """Load model from persistent storage."""
        with open(self.trained_model_path, "rb") as f:
            return f.read()


    def SendActions(self, request, context):
        """Client use this service to send trajectories and server use this to receive
        It is client's job to make sure send enough trajectories to sever for training"""
        if len(request.actions) <= hyperparam_server['batch_size']:
            return trajectory_pb2.ActionResponse(code=0, message=f"Not enough trajectory for training, at least send {hyperparam_server['batch_size']} trajectories.")
        print(f"Received {len(request.actions)} actions for training from client.")

        actions = self._see_actions(request.actions)

        # clean model_ready before start 
        self.model_ready = 0
        response = threading.Thread(target=self._algorithm.receive_trajectory, args=(actions)).start() # start training
        return trajectory_pb2.ActionResponse(code=1, message=f"Training started successfully for client.")

    def ClientPoll(self, request, context):
        """
        parameters:
            request: Empty. Client don't send any info to server on this request
            context: gRPC transmission context
        Return:
            a RL4SysModel proto3 obj. include three fields:
            code: 1, 0, -1. 1 means all good. 0 means still training. -1 means error.
        """
        # Server received a poll from unknown server
        print(f"[Client Poll] Received poll request from client...")
        timeout = 0
        while True:
            if timeout >= self.idle_timeout:
                return trajectory_pb2.RL4SysModel(code=0, model=b"", error="Model is still training.")

            with self.lock:
                # Initial handshake
                if request.first_time == 1:
                    print(f"Client model version {request.version}, Server model version ") # TODO model need to have version scheme
                    print(f"[Client Poll] Handshake initiated by client.")
                    return trajectory_pb2.RL4SysModel(code=1, model=b"", version=0, error="Handshake successful.")

                if self.model_ready == 1:
                    print(f"[Client Poll] Model is ready for client. Sending model.")
                    model_data = self.trained_model
                    return trajectory_pb2.RL4SysModel(code=1, model=model_data, error="")
                elif self.model_ready == -1:
                    print(f"[Client Poll] Error for client: {self.error_message}")
                    return trajectory_pb2.RL4SysModel(code=-1, model=b"", error=self.error_message)
            
            time.sleep(1)
            timeout += 1

    def _get_actions(self, actions, verbose = False):
        """This function deserialize tensors from byte and return a list of actions"""
        actions = []
        for action in actions:
            action = deserialize_action(action)
            actions.append(action)

            if verbose:
                print("Deserialized Action Fields:")
                print(f"  obs: {action.obs_tensor}, {type(action.obs_tensor)}")
                print(f"  action: {action.action_tensor}, {type(action.action_tensor)}")
                print(f"  mask: {action.mask_tensor}, {type(action.mask_tensor)}")
                print(f"  reward: {action.reward}")
                print(f"  data: {action.data}")
                print(f"  done: {action.done}")
                print(f"  reward_update_flag: {action.reward_update_flag}")
        
        return actions
    
def start_training_server(algorithm_name: str,
                         input_size: int,
                         action_dim: int,
                         hyperparams: list[str] | dict,
                         env_dir: str = os.getcwd(),
                         tensorboard: bool = False):
    """
    Creates and starts the TrainingServer, which serves the RL4SysRoute gRPC service.

    Args:
        algorithm_name (str): Name of the algorithm class located in RL4Sys/algorithms.
        input_size (int): Observation dimension or input features to the model.
        action_dim (int): Number of possible actions the model can take.
        hyperparams (list[str] | dict): Either a list of command-line arguments or a dict
                                        of hyperparameters to initialize the model.
        env_dir (str): Directory for environment or logging, defaults to current directory.
        tensorboard (bool): Whether to use Tensorboard logging.
    """
    # 1. Create the gRPC server with a thread pool
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # 2. Instantiate your TrainingServer class
    training_server = TrainingServer(
        algorithm_name=algorithm_name,
        input_size=input_size,
        action_dim=action_dim,
        hyperparams=hyperparams,
        env_dir=env_dir,
        tensorboard=tensorboard
    )

    # 3. Add the servicer to the server
    trajectory_pb2_grpc.add_RL4SysRouteServicer_to_server(training_server, server)

    # 4. Bind the server to a port
    server.add_insecure_port('[::]:50051')

    # 5. Start the server
    server.start()
    print("Training Server started, listening on port 50051.")

    # 6. Block until the server is terminated
    server.wait_for_termination()

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="RL4Sys Training Server",
        description="Starts the RL4Sys training server with the specified algorithm and parameters.",
        epilog=(
            "Pass algorithm-specific parameters as additional arguments:\n"
            "  e.g. --buf_size=5 --gamma=.65"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "algorithm",
        type=str,
        help="Name of algorithm to use, as found in RL4Sys/algorithms"
    )
    parser.add_argument(
        "--kernel_size", "--size",
        type=int,
        required=True,
        help="Number of actions in each observation"
    )
    parser.add_argument(
        "--kernel_dim", "--dim",
        type=int,
        required=True,
        help="Number of features per action"
    )

    # Optional argument for Tensorboard
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=False,
        help="Enable Tensorboard logging"
    )

    # Parse known args; everything else goes into `extras` for algorithm hyperparams
    args, extras = parser.parse_known_args()

    # Start the gRPC training server
    start_training_server(
        algorithm_name=args.algorithm,
        input_size=args.kernel_size,
        action_dim=args.kernel_dim,
        hyperparams=extras,
        env_dir=os.getcwd(),
        tensorboard=args.tensorboard
    )