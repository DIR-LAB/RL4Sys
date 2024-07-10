from torch.utils.tensorboard import SummaryWriter

import pickle
import zmq
import threading

from utils.plot import plot_data

import os
import json
""" Import and load RL4Sys/config.json server configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
CONFIG_PATH = os.path.join(top_dir, 'config.json')
monitor_server = {}
monitor_params = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        monitor_server = config['server']
        monitor_server = monitor_server['monitor_server']
        monitor_params = config['utils']
        monitor_params = monitor_params['monitor']
        monitor_params['log_dir'] = os.path.join(monitor_params['log_dir'])
except (FileNotFoundError, KeyError):
    print(f"[Monitor] Failed to load configuration from {CONFIG_PATH}, loading defaults.")
    monitor_server = {
        'prefix': 'tcp://',
        'host': 'localhost',
        'port': ":6000"
    }
    monitor_params = {
        'log_dir': os.path.join(top_dir, 'utils/runs'),
        'filename_suffix': '_board'
    }


"""
In order to monitor the training process as well as the training results, we must implement a monitor class that will
retrieve logged data and display it using plots and tables in a tensorboard instance. 
"""

monitor_address = f"{monitor_server['prefix']}{monitor_server['host']}{monitor_server['port']}"


def send_data_to_monitor(data: dict) -> None:
    """

    :param data:
    :return:
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(monitor_address)
    socket.send_string("data")
    socket.send(pickle.dumps(data))
    socket.close()
    context.term()


def send_step_to_monitor(step: list[tuple[str, int]]) -> None:
    """

    :param step:
    :return:
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(monitor_address)
    socket.send_string("step")
    socket.send(pickle.dumps(step))
    socket.close()
    context.term()


def check_for_monitor_server() -> bool:
    """

    :return:
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(monitor_address)
    socket.setsockopt(zmq.RCVTIMEO, 5000)

    try:
        socket.send_string("ping")
        message = socket.recv_string()
        if message == "pong":
            return True
    except Exception as e:
        print(f"[Monitor] Error: {e}")
    finally:
        socket.close()
        context.term()

    return False


class Monitor:
    """

    """
    def __init__(self, log_dir=str(monitor_params['log_dir']), purge_step=None, max_queue=10, flush_secs=120,
                 filename_suffix=str(monitor_params['filename_suffix'])):
        self.writer = SummaryWriter(log_dir=log_dir, purge_step=purge_step, max_queue=max_queue,
                                    flush_secs=flush_secs, filename_suffix=filename_suffix)
        self.data = dict()
        self.global_steps = list[tuple()]

        self._listen_thread = threading.Thread(target=self._listening_loop)
        self._listen_thread.daemon = True
        self._listen_thread.start()

        print("[Monitor] Initialized")

    def _listening_loop(self):
        """

        :return:
        """
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind(monitor_address)

        while True:
            message = socket.recv_string()
            if message == b'data':
                data = socket.recv()
                self.data = pickle.loads(data)
            elif message == b'step':
                step = socket.recv()
                self.global_steps = pickle.loads(step)

        socket.close()
        context.term()

    def display_plot(self, xaxis, value, condition, smooth, **kwargs):
        """

        :return:
        """
        if self.data:
            plot_data(self.data, xaxis, value, condition, smooth, **kwargs)

    def _start_tensorboard(self):
        """

        :return:
        """
        if not os.path.exists(self.writer.log_dir):
            print("[Monitor] Directory not found. Tensorboard not started.")
            return

        import subprocess
        try:
            print("[Monitor] Starting Tensorboard.")
            subprocess.run(["tensorboard", "--logdir", monitor_params['log_dir']])
        except Exception as e:
            print(f"[Monitor] Error: {e}")

    def start_tensorboard_in_thread(self):
        """

        :return:
        """
        tensorboard_thread = threading.Thread(target=self._start_tensorboard)
        tensorboard_thread.daemon = True
        tensorboard_thread.start()



"""
Below runs the monitor class with the aim of proving PPO results consistent with the
RLScheduler paper's results {https://arxiv.org/abs/1910.08925}. 
"""

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Monitor training results.')
