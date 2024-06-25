from action import RL4SysAction
import zmq
import pickle

import os

import json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
max_traj_length = None
traj_server = None
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        max_traj_length = config['max_traj_length']
        traj_server = config['server']
        traj_server = traj_server['trajectory_server']
except (FileNotFoundError, KeyError):
    print(f"Failed to load configuration from {CONFIG_PATH}, loading defaults.")
    max_traj_length = 1000
    traj_server = {
        'prefix': 'tcp://',
        'host': 'localhost',
        'port': 5555
    }

class RL4SysTrajectory:
    """Container for trajectories in RL4Sys environments.

    Stores actions taken in environment script to send to training_server.
    Transfers data to training server as needed.

    Attributes:
        max_length (int): Ignored in current implementation.
        actions (list of RL4SysAction): List of actions taken.

    """

    def __init__(self, max_length: int = max_traj_length):

        if max_length:
            self.max_length = max_length
        else:
            self.max_length = max_traj_length

        self.actions: list[RL4SysAction] = []

    def add_action(self, action: RL4SysAction) -> None:
        """Add action to trajectory.

        If action.done is true, will send trajectory to training server and clear local copy.

        Args:
            action: the action to add to trajectory. Should have property done = True if this trajectory has finished.

        """
        self.actions.append(action)

        if action.done:
            print("[trajectory.py - whole traj - send to Training Server]")

            # TODO refactor out to RL4SysAgent object which holds this trajectory, or allow connection information to be passed in from agent
            send_trajectory(self)
            self.actions = [] # reset the trajectory
            if len(self.actions) >= self.max_length:
                print("traj too long, ignored in current implementation") # TODO handle max traj length

def serialize_trajectory(trajectory: RL4SysTrajectory) -> bytes:
    """Pickle trajectory.

    Used to send trajectory over network.
    Unpickle with pickle.loads(bytes).

    Args:
        trajectory: the trajectory to serialize.
    Returns:
        Pickled trajectory object.

    """
    return pickle.dumps(trajectory)

def send_trajectory(trajectory: RL4SysTrajectory) -> None:
    """Send trajectory over network.

    Currently uses tcp://localhost:5555.

    Args:
        trajectory: the trajectory to send.

    """
    # Serialize the trajectory
    serialized_trajectory = serialize_trajectory(trajectory)

    # Create a ZMQ context and a push socket
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    # Assuming the server is on localhost, port 5555
    socket.connect(f"{traj_server['prefix']}{traj_server['host']}:{traj_server['port']}")

    # Send the trajectory data
    socket.send(serialized_trajectory)

    # Close the socket and context
    socket.close()
    context.term()