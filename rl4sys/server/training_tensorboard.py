from torch.utils.tensorboard import SummaryWriter
from utils.plot import get_datasets, get_newest_dataset

import time
import queue
import threading

import os
import json

import pandas as pd
from utils.conf_loader import ConfigLoader

""" Import and load RL4Sys/config.json asynchronous tensorboard parameters and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
config_loader = ConfigLoader()
tb_params = config_loader.tb_params


class TensorboardWriter:
    """Synchronous Tensorboard writer:
    Creates a single thread that writes data to tensorboard, and for running the tensorboard.
    Writes scalar values to tensorboard based on input x-axis step value and n number of y-values according to
    queue element output sequence.

    Can only write scalars to tensorboard (for now).

    Manually launch_tensorboard(logdir) or,
    cmd:
        tensorboard --logdir <path_to_tensorboard_logs>

    """

    def __init__(self, scalar_tags=tb_params['scalar_tags'], max_count_per_scalar=tb_params['max_count_per_scalar'],
                 global_step_tag=tb_params['global_step_tag'], env_dir=os.getcwd(), algorithm_name: str = 'run'):
        self.writer = None
        self._data_log_dir = env_dir + '/logs'
        self._file_root = get_newest_dataset(self._data_log_dir, return_file_root=True)
        self._tb_log_dir = self._file_root + f'/tb_' + algorithm_name.lower() + f'_{int(time.time())}'
        self._file = self._file_root + '/progress.txt'

        self.data_queue = queue.Queue()

        self.valid_tags = False
        self.scalar_tags = scalar_tags.split(';')
        self._max_count_per_scalar = max_count_per_scalar
        self._total_scalar_count = len(self.scalar_tags) * self._max_count_per_scalar
        self._global_step_tag = global_step_tag
        self._recent_global_step = 0

        self._loop_stop_signal = threading.Event() # make sure to set this on app sim
        self._tb_thread = threading.Thread(target=self._tensorboard_writer_processes)
        self._tb_thread.daemon = True # edit to 'True' for end properly
        self._tb_thread.start()

        print("[TensorboardWriter] Initialized")

    def manually_queue_scalar(self, tag: str, scalar_value: float, global_step: int):
        self.data_queue.put(('scalar', tag, scalar_value, global_step))

    def _tensorboard_writer_processes(self):
        """
        Main loop for tensorboard writer thread.
        """
        def _validate_tag_existence():
            """
            Validates that the scalar tags exist in the progress.txt file.
            """
            print("[TensorboardWriter - _validate_tag_existence] Validating scalar tags...")

            if not os.path.exists(self._file_root):
                print("[TensorboardWriter - _validate_tag_existence] Data directory not found. Tensorboard not started.")
                return False

            if not os.path.exists(self._file):
                print("[TensorboardWriter - _validate_tag_existence] progress.txt not found. Tensorboard not started.")
                return False

            if os.path.getsize(self._file) == 0:
                print("[TensorboardWriter - _validate_tag_existence] progress.txt is empty. Tensorboard not started.")
                return False

            data = pd.read_table(self._file)
            if not data.empty:
                for scalar in self.scalar_tags:
                    if scalar not in data.columns:
                        print(f"[TensorboardWriter - _validate_tag_existence] {scalar} not found in progress.txt. "
                              f"Removing from scalar tags.")
                        self.scalar_tags.remove(scalar)
                        self._total_scalar_count -= self._max_count_per_scalar
                if not self.scalar_tags:
                    print("[TensorboardWriter - _validate_tag_existence] No scalar tags found. Tensorboard not started.")
                    self.valid_tags = False
                    return None
                else:
                    self.valid_tags = True
                    return None
            else:
                print("[TensorboardWriter - _validate_tag_existence] Data is empty. Tensorboard not started.")
                return False

        def _retrieve_and_queue_data(_previous_last_step: int):
            """
            retrieves data from progress.txt and queues it for writing.

            queues according to order of scalar tags per step.
            i.e. Step 1 -> AverageEpRet, StdEpRet
                 Step 2 -> AverageEpRet, StdEpRet
                 ...
            """
            print("[TensorboardWriter - _retrieve_and_queue_data] Retrieving data from progress.txt...")

            if not os.path.exists(self._file_root):
                print("[TensorboardWriter - _retrieve_and_queue_data] Data directory not found. Tensorboard not "
                      "started.")
                return 0, 0

            if not os.path.exists(self._file):
                print("[TensorboardWriter - _retrieve_and_queue_data] progress.txt not found. Tensorboard not started.")
                return 0, 0

            if os.path.getsize(self._file) == 0:
                print("[TensorboardWriter - _retrieve_and_queue_data] progress.txt is empty. Tensorboard not started.")
                return 0, 0

            data = pd.read_table(self._file)
            _queued_count = 0
            if not data.empty:
                new_last_step = int(data[self._global_step_tag].idxmax())
                for scalar in self.scalar_tags:
                    for i in range(_previous_last_step + 1, new_last_step + 1):
                        self.data_queue.put(('scalar', scalar, data[scalar][i], data[self._global_step_tag][i]))
                        _queued_count += 1
                return new_last_step, _queued_count
            else:
                print("[TensorboardWriter - _retrieve_and_queue_data] Data is empty. Tensorboard not started.")
                return 0, 0

        # processes main loop below
        while True:
            if _validate_tag_existence() is not None:
                time.sleep(5)
            else:
                break
        if not self.valid_tags:
            print("[TensorboardWriter - _tensorboard_processes] No valid tags found. Stopping.")
            self._loop_stop_signal.set()
            return
        else:
            self.writer = SummaryWriter(log_dir=self._tb_log_dir, filename_suffix='_tb')
            previous_last_step = 0
            while not self._loop_stop_signal.is_set():
                previous_last_step, queued_count = _retrieve_and_queue_data(previous_last_step)
                if queued_count != 0:
                    try:
                        for count in range(queued_count):
                            write_type, *args = self.data_queue.get()
                            if write_type == 'scalar':
                                tag, scalar_value, global_step = args
                                if self._total_scalar_count > 0:
                                    print(f"[TensorboardWriter - _tensorboard_processes] Writing scalar: {args}")
                                    self.writer.add_scalar(tag, scalar_value, global_step)
                                    self._total_scalar_count -= 1
                        self.writer.flush()
                    except queue.Empty:
                        continue
                    finally: # TODO why writer have limit
                        if self._total_scalar_count <= 0:
                            print("[TensorboardWriter - _tensorboard_processes] Max scalar count per tag reached. Stopping.")
                            self._loop_stop_signal.set()
                else:
                    time.sleep(10)
            self.writer.close()
            # self._tb_thread.join() # ??? why join itself, dead lock
            return


def launch_tensorboard(logdir: str):
    """
        Launches tensorboard process in the background.
        Uses tb_log_dir parameter from config.json for log pathing.
    :return:
    """
    import subprocess
    try:
        print("[launch_tensorboard] Starting Tensorboard.")
        top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../RL4Sys'))
        subprocess.run(["tensorboard", "--logdir", logdir])
    except Exception as e:
        print(f"[launch_tensorboard] Error: {e}")
    finally:
        print("[launch_tensorboard] Tensorboard closed.")
