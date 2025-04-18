from abc import ABC, abstractmethod

import pickle

from ...._common._rl4sys.BaseAction import RL4SysActionAbstract

from ....utils.conf_loader import ConfigLoader

config_loader = ConfigLoader()
TRAINING_SERVER_ADDRESS = config_loader.train_server_address


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


# Note: The send_trajectory function has been removed as it used the old ZMQ implementation.
# Trajectories are now sent using gRPC in the client/agent.py implementation.
