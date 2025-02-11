import sys
import os
import time
import threading
import io

import torch
from numpy import ndarray

import grpc
from protocol import trajectory_pb2, trajectory_pb2_grpc
from protocol.trajectory import RL4SysTrajectory
from protocol.action import RL4SysAction

from utils.util import deserialize_model  # or your own full-model or state-dict deserializer
from utils.conf_loader import ConfigLoader

config_loader = ConfigLoader()
TRAINING_SERVER_ADDRESS = "localhost:50051"  # or from config
load_model_path = config_loader.load_model_path  # if you want local fallback

class RL4SysAgent:
    """
    Synchronous gRPC-based Agent that:
      - Handshakes once at startup for an initial model
      - Sends trajectories and polls once after each send
      - No continuous background polling thread
    """

    def __init__(self, 
                 model: torch.nn.Module = None, 
                 server_address: str = TRAINING_SERVER_ADDRESS):
        """
        Args:
            model: an optional PyTorch model. Must have .step(obs, mask) -> (action_ndarray, dict_info).
            server_address: "host:port" of the gRPC training server.
        """
        self.server_address = server_address
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = trajectory_pb2_grpc.RL4SysRouteStub(self.channel)

        self._lock = threading.Lock()
        self._model = model
        self._current_traj = RL4SysTrajectory()

        # If a model was provided, validate it
        if self._model is not None:
            self._validate_model(self._model)

        # 1) Single handshake at init: we attempt to get an initial model
        self.local_version = 0
        self._handshake_for_initial_model()

    def _handshake_for_initial_model(self) -> None:
        """
        One-time handshake with the server to see if it has an initial model.
        If code=1 and model bytes are nonempty, we deserialize it.
        Otherwise, we just note the version or error.
        """
        print("[RL4SysAgent] Handshake: requesting initial model from server...")
        req = trajectory_pb2.RequestModel(first_time=1, version=self.local_version)
        try:
            resp = self.stub.ClientPoll(req)
        except grpc.RpcError as e:
            print(f"[RL4SysAgent] gRPC error during handshake: {e.details()}")
            return

        if resp.code == 1:
            print("[RL4SysAgent] Handshake successful with server.")
            self.local_version = resp.version
            if len(resp.model) > 0:
                with self._lock:
                    self._model = deserialize_model(resp.model)
                print("[RL4SysAgent] Received and loaded initial model from server.")
            else:
                print("[RL4SysAgent] Server has no initial model yet (model bytes empty).")
        elif resp.code == 0:
            print("[RL4SysAgent] Server not ready yet (code=0). No initial model.")
        else:  # code == -1 or other
            print(f"[RL4SysAgent] Handshake error or server error: code={resp.code}, err={resp.error}")

    def _validate_model(self, model: torch.nn.Module) -> None:
        """Check that the model has .step(...) returning (ndarray, dict)."""
        assert hasattr(model, 'step'), "Model must have a .step(...) method."
        result = model.step(None, None)
        assert isinstance(result, tuple), "Model.step(...) must return a tuple."
        assert isinstance(result[0], ndarray), "First element of tuple must be numpy ndarray."
        assert isinstance(result[1], dict), "Second element of tuple must be a dict."

    def request_for_action(self, obs: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> RL4SysAction:
        """
        Produce an action from the current model. Stores the action in our local trajectory buffer.
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("No model available yet!")
            action_nd, data_dict = self._model.step(obs, mask)

        r4sa = RL4SysAction(obs, action_nd, mask, reward=-1, data=data_dict, done=False)
        self._current_traj.add_action(r4sa)
        return r4sa

    def flag_last_action(self, reward: int) -> None:
        """
        Mark the end of the current trajectory, send it to the server, and poll for an updated model.
        """
        # Create terminal action
        last_action = RL4SysAction(obs=None, action=None, mask=None, reward=None,
                                   data=None, done=True)
        last_action.update_reward(reward)
        self._current_traj.add_action(last_action)

        # 1) Send trajectory to server for training
        self._send_trajectory_to_server()
        print("[BaseTrajectory.py - whole traj - send to Training Server]")

        # 2) Poll once for a new model (server may or may not have a fresh one yet)
        self._poll_for_model_update()

        # 3) Clear local trajectory
        self._current_traj = RL4SysTrajectory()

    def _send_trajectory_to_server(self) -> None:
        """
        Builds a gRPC ActionList message from self._current_traj and calls SendActions.
        """
        action_msgs = []
        for action in self._current_traj.actions:
            # Debug print to see what we're working with
            print(f"[Debug] Processing action:")
            print(f"  obs: {type(action.obs)}")
            print(f"  act: {type(action.act)}")
            print(f"  mask: {type(action.mask)}")
            print(f"  rew: {type(action.rew)}")
            print(f"  done: {action.done}")
            print(f"  reward_update_flag: {action.reward_update_flag}")
            print(f"  data: {action.data}")


            action_proto = trajectory_pb2.RL4SysAction(
                obs=action.obs.numpy().tobytes() if action.obs is not None else b"None",
                action=action.act.to_bytes(4, byteorder='big', signed=True) if action.act is not None else b"None",
                mask=action.mask.numpy().tobytes() if action.mask is not None else b"None",
                reward=int(action.rew) if action.rew is not None else 0,
                done=action.done if type(action.done) != None else False,
                reward_update_flag=action.reward_update_flag if type(action.reward_update_flag) != None else False,
                data={str(k): str(v) for k,v in (action.data or {}).items()} if action.data is not None else {"none":"none"}
            )
            print(action_proto.obs)
            print(action_proto.action)
            print(action_proto.mask)
            print(action_proto.reward)
            print(action_proto.done)
            print(action_proto.reward_update_flag)
            print(action_proto.data)


            action_msgs.append(action_proto)

        action_list = trajectory_pb2.RL4SysActionList(actions=action_msgs)
        print(action_list)

        try:
            response = self.stub.SendActions(action_list)
            if response.code == 1:
                print(f"[RL4SysAgent] Successfully sent trajectory: {response.message}")
            else:
                print(f"[RL4SysAgent] Server rejected trajectory: {response.message}")
        except grpc.RpcError as e:
            print(f"[RL4SysAgent] gRPC error sending trajectory: {e.details()}")

    def _poll_for_model_update(self) -> None:
        """
        Makes a single gRPC call to ClientPoll to see if the server has a new model.
        If code=1 and we get non-empty model bytes, we load it.
        Otherwise we do nothing further.
        """
        poll_req = trajectory_pb2.RequestModel(first_time=0, version=self.local_version)
        try:
            poll_resp = self.stub.ClientPoll(poll_req)
        except grpc.RpcError as e:
            print(f"[RL4SysAgent] gRPC error while polling for model: {e.details()}")
            return

        if poll_resp.code == 1:
            # Possibly a new model or the same version
            if len(poll_resp.model) > 0:
                with self._lock:
                    self._model = deserialize_model(poll_resp.model)
                self.local_version = poll_resp.version
                print("[RL4SysAgent] Updated local model from server (poll).")
            else:
                # code=1 but no new model bytes => server has no newer version
                pass
        elif poll_resp.code == 0:
            # Model not ready or server is still training
            pass
        elif poll_resp.code == -1:
            print(f"[RL4SysAgent] Server reported error: {poll_resp.error}")

    def close(self):
        """Cleanly close the gRPC channel if needed."""
        self.channel.close()
        print("[RL4SysAgent] Closed gRPC channel.")
