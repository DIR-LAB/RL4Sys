import grpc
import trajectory_pb2
import trajectory_pb2_grpc
import io
import torch
import time

# Helper function to simulate creating a trajectory
def create_trajectory():
    return trajectory_pb2.RL4SysAction(
        obs=b"dummy_obs",  # Placeholder for serialized tensor
        action=b"dummy_action",  # Placeholder for serialized tensor
        mask=b"dummy_mask",  # Placeholder for serialized tensor
        reward=10,
        data={"key": "value"},
        done=False,
        reward_update_flag=False,
    )

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = trajectory_pb2_grpc.RL4SysRouteStub(channel)

        # Create and send trajectories
        actions = [create_trajectory() for _ in range(5)]  # Simulate 5 actions
        trajectory = trajectory_pb2.RL4SysActionList(actions=actions)
        response = stub.SendActions(trajectory)
        print(f"Server response: {response.code}, {response.message}")

        # Poll for model readiness
        while True:
            poll_response = stub.ClientPoll(trajectory_pb2.Empty())
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
