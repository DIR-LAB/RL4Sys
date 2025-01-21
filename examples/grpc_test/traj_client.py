import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import grpc
from protocol import trajectory_pb2
from protocol import trajectory_pb2_grpc
import io
import torch
import time
from utils.util import serialize_tensor


# Helper function to simulate creating a trajectory
def create_trajectory():
    # Generate real tensors
    obs_tensor = torch.randn(8,)  # Example: A 3x3 tensor
    action_tensor = torch.tensor(1)  # Example: A scalar tensor representing an action
    mask_tensor = torch.zeros(8,)  # Example: A 1D tensor of size 3

    # Serialize tensors to bytes
    serialized_obs = serialize_tensor(obs_tensor)
    serialized_action = serialize_tensor(action_tensor)
    serialized_mask = serialize_tensor(mask_tensor)

    # Return a Protobuf RL4SysAction object with serialized tensors
    return trajectory_pb2.RL4SysAction(
        obs=serialized_obs,
        action=serialized_action,
        mask=serialized_mask,
        reward=10,  # Example reward
        data={"key": "value"},  # Example metadata
        done=False,
        reward_update_flag=False,
    )

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = trajectory_pb2_grpc.RL4SysRouteStub(channel)

        # Create and send trajectories
        actions = [create_trajectory() for _ in range(5)]  # Simulate 5 actions
        trajectory = trajectory_pb2.RL4SysActionList(actions=actions)
        try:
            response = stub.SendActions(trajectory)
        except grpc.RpcError as e:
            print(f"Error while sending actions to the server: {e.details()}")
            return
        print(f"Server response: {response.code}, {response.message}")

        # Poll for model readiness
        while True:
            try:
                poll_response = stub.ClientPoll(trajectory_pb2.Empty())
            except grpc.RpcError as e:
                print(f"Error while polling the server: {e.details()}")
                break
            if poll_response.code == 1:
                print("Model is ready!")
                # Deserialize the model
                model_buffer = io.BytesIO(poll_response.model)
                state_dict = torch.load(model_buffer)
                model = torch.nn.Linear(10, 5)  # Instantiate the same model architecture
                model.load_state_dict(state_dict)
                print("Loaded model successfully.")
                break
            elif poll_response.code == -1:
                print(f"Error: {poll_response.error}")
                break
            else:
                print("Model is still training. Polling again in 2 seconds...")
                time.sleep(2)

if __name__ == "__main__":
    run()
