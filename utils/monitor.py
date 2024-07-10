import numpy as np
from torch.utils.tensorboard import SummaryWriter

import zmq
import threading

from utils.plot import *
from utils.logger import EpochLogger

import os
import json
""" Import and load RL4Sys/config.json server configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
monitor_server = {}
monitor_params = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        monitor_server = config['server']
        monitor_server = monitor_server['training_server']
        monitor_params = config['utils']
        monitor_params = monitor_params['monitor']
        monitor_params['log_dir'] = os.path.join(os.path.dirname(__file__), monitor_params['log_dir'])
except (FileNotFoundError, KeyError):
    print(f"[Monitor]: Failed to load configuration from {CONFIG_PATH}, loading defaults.]")
    monitor_server = {
        'prefix': 'tcp://',
        'host': 'localhost',
        'port': ":6000"
    }
    monitor_params = {
        'log_dir': os.path.join(os.path.dirname(__file__), 'runs')
        'filename_suffix': '_board'
    }



"""
In order to monitor the training process as well as the training results, we must implement a monitor class that will
retrieve logged data and display it using plots and tables in a tensorboard instance. 
"""


class Monitor:
    """

    """
    def __init__(self, log_dir=monitor_params['log_dir'], purge_step=None, max_queue=10, flush_secs=120,
                 filename_suffix=monitor_params['filename_suffix']):
        self.writer = SummaryWriter(log_dir=log_dir, purge_step=purge_step, max_queue=max_queue,
                                    flush_secs=flush_secs, filename_suffix=filename_suffix)
        self.data = {}
        self.global_step = 0

        self._listen_thread = threading.Thread(target=self._loop_for_step_updates)
        self._listen_thread.daemon = True
        self._listen_thread.start()

        print("[Monitor] Initialized")

    def _loop_for_step_updates(self):
        """

        :return:
        """


    def display_plot(self, xaxis, yaxis, condition, smooth, **kwargs):
        """

        :return:
        """
        pass

    def start_tensorboard(self):
        """

        :return:
        """
        import subprocess
        try:
            print("[Monitor] Starting Tensorboard.")
            subprocess.run(["tensorboard", "--logdir", self.writer.log_dir])
        except FileNotFoundError:
            print("[Monitor] Directory not found. Tensorboard not started.")


"""
The 'main' function in this file runs the monitor class with the aim of proving algorithmic results consistent with the
RLScheduler paper's results {https://arxiv.org/abs/1910.08925}. 
"""

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Monitor training results.')
