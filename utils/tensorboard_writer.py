from torch.utils.tensorboard import SummaryWriter
from utils.plot import get_datasets

import time
import queue
import threading

import os
import json

""" Import and load RL4Sys/config.json asynchronous tensorboard parameters and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
top_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
CONFIG_PATH = os.path.join(top_dir, 'config.json')
tb_server = {}
tb_params = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
        tb_params = config['utils']
        tb_params = tb_params['tensorboard_writer']
except (FileNotFoundError, KeyError):
    print(f"[TensorboardWriter] Failed to load configuration from {CONFIG_PATH}, loading defaults.")
    tb_params = {
        'tb_log_dir': 'utils/runs',
        'filename_suffix': '_board',
        'data_log_dir': 'data',
        'data_refresh_rate': 5000,
        'scalar_tags': 'AverageEpRet;StdEpRet',
        'max_count_per_scalar': 100,
        'global_step_tag': 'Epoch'
    }
finally:
    tb_params['tb_log_dir'] = os.path.join(top_dir, tb_params['tb_log_dir'])
    tb_params['data_log_dir'] = os.path.join(top_dir, tb_params['data_log_dir'])


class TensorboardWriter:
    """Synchronous Tensorboard writer:
    Creates a single thread that writes data to tensorboard, and for running the tensorboard.
    Writes scalar values to tensorboard based on input x-axis step value and n number of y-values according to
    queue element output sequence.

    Can only write scalars to tensorboard (for now).
    """

    def __init__(self, tb_log_dir=str(tb_params['tb_log_dir']), purge_step=None, max_queue=10, flush_secs=120,
                 filename_suffix=str(tb_params['filename_suffix']), data_log_dir=tb_params['data_log_dir'],
                 data_refresh_rate=tb_params['data_refresh_rate'], scalar_tags=tb_params['scalar_tags'],
                 max_count_per_scalar=tb_params['max_count_per_scalar'], global_step_tag=tb_params['global_step_tag']):
        
        self.writer = SummaryWriter(log_dir=tb_log_dir, purge_step=purge_step, max_queue=max_queue,
                                    flush_secs=flush_secs, filename_suffix=filename_suffix)
        self.data = queue.Queue()
        self._data_log_dir = data_log_dir
        self._data_refresh_rate = int(data_refresh_rate)

        self.scalar_tags = scalar_tags.split(';')
        self._scalar_count_bucket = len(self.scalar_tags) * max_count_per_scalar
        self._global_step_tag = global_step_tag
        self._recent_global_step = 0

        self._tb_thread = threading.Thread(target=self._tensorboard_processes)
        self._tb_thread.daemon = False
        self._tb_thread.start()
        self._loop_stop_signal = threading.Event()

        print("[TensorboardWriter] Initialized")

    def manually_queue_scalar(self, tag: str, scalar_value: float, global_step: int):
        self.data.put(('scalar', tag, scalar_value, global_step))

    def _tensorboard_processes(self):
        """
        :return:
        """
        while not self._loop_stop_signal.is_set():
            self._retrieve_and_queue_data()
            time.sleep(self._data_refresh_rate / 1000)  # milliseconds
            try:
                for _ in self.scalar_tags:
                    write_type, *args = self.data.get()
                    if write_type == 'scalar':
                        if self._scalar_count_bucket > 0:
                            print(f"[TensorboardWriter] Writing scalar: {args}")
                            self.writer.add_scalar(*args)
                            self._scalar_count_bucket -= 1
            except queue.Empty:
                continue
            finally:
                if self._scalar_count_bucket <= 0:
                    print("[TensorboardWriter] Max scalar count per tag reached. Stopping.")
                    self._loop_stop_signal.set()
        self.writer.close()
        self._launch_tensorboard()

    def _retrieve_and_queue_data(self):
        """
        retrieves data from progress.txt and queues it for writing.

        queues according to order of scalar tags per step.
        i.e. Step 1 -> AverageEpRet, StdEpRet
             Step 2 -> AverageEpRet, StdEpRet
             ...
        :return:
        """
        print("[TensorboardWriter] Retrieving data from progress.txt.")
        dataset = get_datasets(self._data_log_dir)
        for data in dataset:
            for scalar in self.scalar_tags:
                self.data.put(('scalar', scalar, data[scalar], data[self._global_step_tag]))

    def _launch_tensorboard(self):
        """

        :return:
        """
        if not os.path.exists(self.writer.log_dir):
            print("[TensorboardWriter] Directory not found. Tensorboard not started.")
            return

        import subprocess
        try:
            print("[TensorboardWriter] Starting Tensorboard.")
            subprocess.run(["tensorboard", "--logdir", tb_params['log_dir']])
        except Exception as e:
            print(f"[TensorboardWriter] Error: {e}")
        finally:
            print("[TensorboardWriter] Tensorboard closed.")
            self._tb_thread.join()
