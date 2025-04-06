from abc import ABC, abstractmethod

import pickle
import zmq

from _common._rl4sys.BaseAction import RL4SysActionAbstract

from utils.conf_loader import ConfigLoader

config_loader = ConfigLoader()
traj_server = config_loader.traj_server


class RL4SysTrajectoryAbstract(ABC):
    """
    Abstract class for a trajectory in RL4Sys.
    """
    def __init__(self, *args, **kwargs):
        super(RL4SysTrajectoryAbstract, self).__init__(*args, **kwargs)

    @abstractmethod
    def add_action(self, action: RL4SysActionAbstract):
        """
        Add an action to the trajectory.

        args:
            action: the action to add to the trajectory.
        """
        pass


def serialize_trajectory(trajectory: RL4SysTrajectoryAbstract) -> bytes:
    """Pickle trajectory.

    Used to send trajectory over network.
    Unpickle with pickle.loads(bytes).

    Args:
        trajectory: the trajectory to serialize.
    Returns:
        Pickled trajectory object.

    """
    return pickle.dumps(trajectory)


def send_trajectory(trajectory: RL4SysTrajectoryAbstract) -> None:
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
    address = f"{traj_server['prefix']}{traj_server['host']}{traj_server['port']}"
    socket.connect(address)

    # Send the trajectory data
    socket.send(serialized_trajectory)

    # Close the socket and context
    socket.close()
    context.term()
