from action import RL4SysAction
import zmq
import pickle

MAX_LENGTH = 1000

class RL4SysTrajectory:
    def __init__(self):
        self.max_length = MAX_LENGTH
        self.actions = []

    def add_action(self, action: RL4SysAction):
        self.actions.append(action)

        if action.done:
            #print("[trajectory.py - whole traj - send to Training Server]")

            send_trajectory(self)
            self.actions = [] # reset the trajectory
            # if the traj is too long, do nothing at this moment, keep adding it
            if len(self.actions) >= self.max_length:
                print("traj too long, ignored in current implementation")


def serialize_trajectory(trajectory):
    return pickle.dumps(trajectory)

def send_trajectory(trajectory: RL4SysTrajectory):
    # Serialize the trajectory
    serialized_trajectory = serialize_trajectory(trajectory)

    # Create a ZMQ context and a push socket
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    # Assuming the server is on localhost, port 5555
    socket.connect("tcp://localhost:5555")

    # Send the trajectory data
    socket.send(serialized_trajectory)

    # Close the socket and context
    socket.close()
    context.term()

    

