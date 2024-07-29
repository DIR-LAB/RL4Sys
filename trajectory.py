from action import RL4SysAction
import zmq
import pickle

import os

import json

from conf_loader import ConfigLoader

"""Import and load RL4Sys/config.json trajectory & server configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
config_loader = ConfigLoader()
max_traj_length = config_loader.max_traj_length
traj_server = config_loader.traj_server

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
    address = f"{traj_server['prefix']}{traj_server['host']}{traj_server['port']}"
    socket.connect(address)

    # Send the trajectory data
    socket.send(serialized_trajectory)

    # Close the socket and context
    socket.close()
    context.term()
