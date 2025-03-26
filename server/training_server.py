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
from utils.util import deserialize_action, serialize_model

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



class TrainingServer(trajectory_pb2_grpc.RL4SysRouteServicer):
    """Train a model for a remote agent.

    Sends one untrained model for agent initialization.
    Receives trajectories over the network.
    Trains model based on trajectories.
    Model is sent asynchronously back to agent whenever it is updated with new training.

    Can initalize algorithm object using python dict or command line arguments.
    """
    def __init__(self, algorithm_name: str, input_size: int, action_dim: int, act_limit: float, hyperparams: Union[dict | list[str]],
                 env_dir: str = os.getcwd(), tensorboard: bool = False, seed = 0):
        # super().__init__(algorithm_name, input_size==input_size, hyperparams=hyperparams, env_dir=env_dir)

        # get algorithm class
        algorithm_module: str = ALGORITHMS_PATH + ".{}".format(algorithm_name) + ".{}".format(algorithm_name)
        algorithm_module: importlib.ModuleType = importlib.import_module(algorithm_module)
        algorithm_class = getattr(algorithm_module, algorithm_name)

        self.algorithm_name = algorithm_name
        config_loader = ConfigLoader(algorithm=algorithm_name)
        self.save_model_path = config_loader.save_model_path
        self.hyperparam_server = config_loader.algorithm_params
        
        # if hyperparams are a list of command-line arguments, parse into a hyperparam dict
        if isinstance(hyperparams, list):
            parser = argparse.ArgumentParser()

            # add each parameter of algorithm class
            parameters = inspect.signature(algorithm_class.__init__).parameters
            no_parse = ('env_dir', 'input_size', 'act_dim', 'act_limit', 'self')
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
        hyperparams['act_limit'] = act_limit

        # instantiate algorithm class
        self._algorithm = algorithm_class(**hyperparams)

        # for server components
        self.idle_timeout = 5 # TODO do it in config loader
        self.lock = threading.Lock()
        self.model_ready = 0
        self.trained_model_path = os.path.join(os.path.dirname(__file__), self.save_model_path, f"{algorithm_name}_model.pth")
        self.error_message = None

        # for trajectory buffer, if trajectory is not enough, we will keep it in buffer
        self.trajectory_buffer = []

        if tensorboard:
            self._tensorboard = TensorboardWriter(env_dir=env_dir, algorithm_name=algorithm_name)

        print("[TrainingServer] Finish Initilizating, Sending the model...")


    def save_model(self, model_data):
        """Save model to persistent storage."""
        os.makedirs(os.path.join(os.path.dirname(__file__), self.save_model_path), exist_ok=True)  # Ensure the directory exists
        with open(self.trained_model_path, "wb") as f:
            f.write(model_data)

    def load_model(self):
        """Load model from persistent storage."""
        with open(self.trained_model_path, "rb") as f:
            return f.read()


    def SendActions(self, request, context):
        """
        Client uses this gRPC method to send trajectories; the server starts training.
        """
        # add trajectory to buffer and check if enough for training
        self.trajectory_buffer.extend(request.actions)
        
        print(f"Received {len(request.actions)} actions for training from client.")

        # Convert proto actions to your local RL4SysAction
        actions = self._get_actions(request.actions, verbose=False)

        # 1) Clear out any previous model signals
        self.model_ready = 0

        # 2) Start a background thread to do the training
        def training_worker():
            updated = self._algorithm.receive_trajectory(actions)
            print(f"[TrainingServer] Training worker finished. updated: {updated}")
            # 'updated' is a boolean indicating if training actually happened
            if updated:
                # If we have a new trained model, store it so that ClientPoll can pick it up
                self.model_ready = 0  # in case the actual training is not instant

                # Either serialize directly to memory, or use self._algorithm.save(...)
                # Example: direct in-memory approach:

                # Now we indicate the model is good to go
                self.model_ready = 1

                # clear trajectory buffer
                self.trajectory_buffer = []
            else:
                # If no epoch triggered, no new model
                self.model_ready = -1
                self.error_message = "No new model trained, please collect more trajectory (which shouldn't happen cause we deal with it before)."
            
            

        # Launch the worker
        thread = threading.Thread(target=training_worker)
        thread.daemon = True
        thread.start()

        return trajectory_pb2.ActionResponse(code=1, message="Training started successfully for client.")

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

            print(f"[Client Poll] Waiting for model update...")
            with self.lock:
                # Initial handshake
                if request.first_time == 1:
                    print(f"Client model version {request.version}, Server model version ") # TODO model need to have version scheme
                    print(f"[Client Poll] Handshake initiated by client.")

                    model_data = None
                    model_critic_data = None
                    if self.algorithm_name == "DQN":
                        model_data = serialize_model(self._algorithm._model)
                    # add for models have two networks
                    elif self.algorithm_name == "PPO":
                        model_data = serialize_model(self._algorithm._model_train.actor)
                        model_critic_data = serialize_model(self._algorithm._model_train.critic)    
                    elif self.algorithm_name == "DDPG":
                        model_data = serialize_model(self._algorithm.ac.actor)
                        model_critic_data = serialize_model(self._algorithm.ac.critic)

                    return trajectory_pb2.RL4SysModel(code=1, model=model_data, model_critic=model_critic_data, version=0, error="Handshake successful.")

                if self.model_ready == 1:
                    print(f"[Client Poll] Model is ready for client. Sending model.")

                    model_data = None
                    model_critic_data = None
                    if self.algorithm_name == "DQN":
                        model_data = serialize_model(self._algorithm._model)
                        return trajectory_pb2.RL4SysModel(code=1, model=model_data, error="")
                    # add for models have two networks
                    elif self.algorithm_name == "PPO":
                        model_data = serialize_model(self._algorithm._model_train.actor)
                        model_critic_data = serialize_model(self._algorithm._model_train.critic)
                    elif self.algorithm_name == "DDPG":
                        model_data = serialize_model(self._algorithm.ac.actor)
                        model_critic_data = serialize_model(self._algorithm.ac.critic)

                    return trajectory_pb2.RL4SysModel(code=1, model=model_data, model_critic=model_critic_data, error="")
                elif self.model_ready == -1:
                    print(f"[Client Poll] Error for client: {self.error_message}")
                    return trajectory_pb2.RL4SysModel(code=-1, model=b"", model_critic=b"", error=self.error_message)
            
            interval = 0.5
            time.sleep(interval) # hyper TODO
            timeout += interval
            if timeout >= self.idle_timeout:
                return trajectory_pb2.RL4SysModel(code=0, model=b"", error="Model is still training.")

    def _get_actions(self, actions, verbose = False):
        """This function deserialize tensors from byte and return a list of actions"""
        deserialized_actions = []  # New list for storing deserialized actions
        for action in actions:
            action = deserialize_action(action)
            deserialized_actions.append(action)

            if verbose:
                print("Deserialized Action Fields:")
                print(f"  obs: {action.obs}, {type(action.obs)}")
                print(f"  action: {action.act}, {type(action.act)}")
                print(f"  mask: {action.mask}, {type(action.mask)}")
                print(f"  reward: {action.rew}")
                print(f"  data: {action.data}")
                print(f"  done: {action.done}")
                print(f"  reward_update_flag: {action.reward_update_flag}")
        
        return deserialized_actions
    
def start_training_server(algorithm_name: str,
                         input_size: int,
                         action_dim: int,
                         act_limit: float,
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

    if algorithm_name == "DDPG" or algorithm_name == "TD3" or algorithm_name == "RPO":
        act_limit = act_limit
    else:   
        act_limit = 1.0

    # 2. Instantiate your TrainingServer class
    training_server = TrainingServer(
        algorithm_name=algorithm_name,
        input_size=input_size,
        action_dim=action_dim,
        act_limit=act_limit,
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




## --------------------------Test Script--------------------------
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
    parser.add_argument("algorithm", type=str, help="Name of algorithm to use, as found in RL4Sys/algorithms")
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