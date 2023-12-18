from .action import RL4SysAction
import zmq
import json

MAX_LENGTH = 1000

class RL4SysTrajectory:
    def __init__(self):
        self.max_length = MAX_LENGTH
        self.actions = []

    def add_action(self, action: RL4SysAction):
        self.actions.append(action)

        if action.done:
            send_trajectory(self)
            self.actions = [] # reset the trajectory
            # if the traj is too long, do nothing at this moment, keep adding it
            if len(self.actions) >= self.max_length:
                print("traj too long, ignored in current implementation")

def send_trajectory(trajectory: RL4SysTrajectory):
    # Serialize the trajectory to JSON
    trajectory_data = json.dumps(trajectory, default=lambda o: o.__dict__)

    # Create a ZMQ context and a push socket
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    # Assuming the server is on localhost, port 5555
    socket.connect("tcp://localhost:5555")

    # Send the trajectory data
    socket.send_string(trajectory_data)

    # Close the socket and context
    socket.close()
    context.term()

    

