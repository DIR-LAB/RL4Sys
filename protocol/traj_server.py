import time
import grpc
from concurrent import futures
import trajectory_pb2
import trajectory_pb2_grpc
import threading
import io
import torch

class RL4SysRouteServicer(trajectory_pb2_grpc.RL4SysRouteServicer):
    def __init__(self):
        self.model_ready = False
        self.trained_model = None
        self.error_message = None
        self.lock = threading.Lock()

    def SendActions(self, request, context):
        print(f"Received {len(request.actions)} actions for training.")

        def simulate_training():
            try:
                time.sleep(5)
                model = torch.nn.Linear(10, 5)
                model_buffer = io.BytesIO()
                torch.save(model.state_dict(), model_buffer)
                with self.lock:  # Protect shared state
                    self.trained_model = model_buffer.getvalue()
                    self.model_ready = True
            except Exception as e:
                with self.lock:
                    self.model_ready = -1
                    self.error_message = str(e)

        threading.Thread(target=simulate_training).start()
        return trajectory_pb2.ActionResponse(code=1, message="Training started successfully.")

    def ClientPoll(self, request, context):
        with self.lock:  # Protect shared state
            if self.model_ready == True:
                return trajectory_pb2.RL4SysModel(code=1, model=self.trained_model, error="")
            elif self.model_ready == -1:
                return trajectory_pb2.RL4SysModel(code=-1, model=b"", error=self.error_message)
            else:
                return trajectory_pb2.RL4SysModel(code=0, model=b"", error="Model is still training.")
            
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    trajectory_pb2_grpc.add_RL4SysRouteServicer_to_server(RL4SysRouteServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started, listening on port 50051.")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
