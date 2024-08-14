from abc import ABC, abstractmethod

import pickle
import zmq

from _common._rl4sys.BaseAction import ActionAbstract

from conf_loader import ConfigLoader

config_loader = ConfigLoader()
traj_server = config_loader.traj_server


class TrajectoryAbstract(ABC):
    def __init__(self, *args, **kwargs):
        super(TrajectoryAbstract, self).__init__(*args, **kwargs)

    @abstractmethod
    def add_action(self, action: ActionAbstract):
        pass


def serialize_trajectory(trajectory: TrajectoryAbstract) -> bytes:
    """Pickle trajectory.

    Used to send trajectory over network.
    Unpickle with pickle.loads(bytes).

    Args:
        trajectory: the trajectory to serialize.
    Returns:
        Pickled trajectory object.

    """
    return pickle.dumps(trajectory)


def send_trajectory(trajectory: TrajectoryAbstract) -> None:
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
