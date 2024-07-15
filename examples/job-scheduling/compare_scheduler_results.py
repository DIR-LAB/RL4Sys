import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import subprocess
import threading

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.plot import get_newest_dataset, plot_data, get_datasets

"""
This script trains models and recreates the plots from the RLScheduler paper.
"""

""" Use Instructions:
----------------------------------------------------------------------------------------------------
First, train a model for each necessary performance metric. Use the following datasets:
    Lublin-1, HPC2N, SDSC-SP2, Lublin-2
Second, after training and logging progress.txt data, plot the results.
Third, personally compare results to RLScheduler paper: {https://arxiv.org/abs/1910.08925}.
"""

_performance_metrics = ['bsld', 'wt', 'tt', 'ru', 'sld']
datasets = ['data/lublin_1024.swf', 'data/HPC2N-2002-2.2-cln.swf', 'data/SDSC-SP2-1998-4.2-cln.swf',
            'data/lublin_256.swf']


def train_scheduler_models(max_epochs: int = 100) -> dict:
    """
    Train all models for each performance metric and/or backfilling option from the RLScheduler paper.
    :return:
    """

    def _run_training_subprocess(workload: str, job_score_type: int, backfil: bool, start_server: str = 'PPO',
                                 num_iterations: int = 100, seed: int = 0):
        """
        Run a training subprocess with the specified parameters.
        """
        command = ['python', './scheduler.py']
        task_args = ['--workload={}'.format(workload), '--seed={}'.format(seed),
                     '--job_score_type={}'.format(job_score_type), '--backfil={}'.format(backfil),
                     '--start-server={}'.format(start_server), '--number-of-iterations={}'.format(num_iterations)]
        command = command + task_args
        file_dir = 'logs/rl4sys-{}-scheduler'.format(str.lower(start_server))

        # check if the final epoch has been reached
        def _check_for_final_epoch(process: subprocess.Popen):
            """
            Check if the final epoch has been reached.
            """
            first_iter = True
            newest_file_address = None
            while True:
                if process is not None:
                    if first_iter:
                        try:
                            newest_file_address = get_newest_dataset(file_dir, return_file_root=True) + '/progress.txt'
                            first_iter = False
                        except Exception as e:
                            print('[compare_scheduler_results.py - CheckForFinalEpoch() - latest_file_dir] '
                                  'Exception: {}'.format(e))
                        time.sleep(180)
                    else:
                        try:
                            latest_data = pd.read_table(newest_file_address)
                            if not latest_data.empty:
                                if latest_data['Epoch'].max() >= max_epochs:
                                    print('[compare_scheduler_results.py - CheckForFinalEpoch() - latest_data] '
                                          'Maximum epoch reached.')
                                    break
                        except Exception as e:
                            print('[compare_scheduler_results.py - CheckForFinalEpoch() - latest_data] '
                                  'Exception: {}'.format(e))
                    time.sleep(60)
            proc.terminate()

        print('Starting subprocess: [{}]'.format(' '.join(command)))
        time.sleep(5)
        proc = subprocess.Popen(command, text=True)
        try:
            # start a thread to check if the final epoch has been reached
            _epoch_thread = threading.Thread(target=_check_for_final_epoch(process=proc))
            _epoch_thread.daemon = True
            _epoch_thread.start()
            _epoch_thread.join()
        except Exception as e:
            print('[compare_scheduler_results.py] Exception: {}'.format(e))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
                print('[compare_scheduler_results.py] Subprocess killed.')
            time.sleep(3)
            log_file = get_newest_dataset(file_dir, return_file_root=True) + '/progress.txt'
            if log_file is None:
                print('[compare_scheduler_results.py] Training subprocess failed.')
                return None
        return log_file

    # train models (train_scheduler_models main loop)
    logs = {}
    category = 1
    while category <= 2:
        backfil = False if category == 1 else True
        for dataset in datasets:
            for metric in _performance_metrics:
                # train models according to parameters
                file = _run_training_subprocess(dataset, _performance_metrics.index(metric), backfil)
                key = '{}_{}_{}'.format(dataset, metric, 'backfil' if backfil else 'nobackfil')
                logs[key] = file
        category += 1

    return logs


def _train_and_plot_models(max_epochs: int = 100) -> None:
    """
    Train and plot models for the RLScheduler paper.
    """
    def _categorize_logs(logs: dict) -> dict:
        """
        Categorize the logs by performance metric and backfill status.
        """
        logs = {}
        for key, data in logs.items():
            # Determine the category and backfill status from the model_key
            _backfil_status = "nobackfil" if "nobackfil" in model_key else "backfil"
            for _dataset in datasets:
                if _dataset in key:
                    for _metric in _performance_metrics:
                        if _metric in model_key:
                            # Create a unique key for each category and backfill status
                            unique_key = f"{_metric}_{_backfil_status}_{_dataset}"
                            if unique_key not in categorized_logs:
                                categorized_logs[unique_key] = {}
                            categorized_logs[unique_key][model_key] = data
                            break

        # Initialize dictionaries for each combination of performance metric and backfill status
        log_categories = {
            f"{metric}_{status}": {} for metric in _performance_metrics for status in ["backfil", "nobackfil"]
        }

        # Iterate over categorized logs and populate the corresponding dictionary
        for key, data in categorized_logs.items():
            for category in log_categories.keys():
                if category in key:
                    log_categories[category][key] = data
                    break
            else:
                print(f'[compare_scheduler_results.py] Invalid log key: {key}')
            return log_categories

    model_logs = train_scheduler_models(max_epochs)
    if not model_logs:
        print('[compare_scheduler_results.py] No models to plot...')
        return

    # categorize data
    categorized_logs = _categorize_logs(model_logs)

    # Plotting the data for each backfil and metric category
    save_address = 'logs/rl4sys-ppo-scheduler/comparative-rlscheduler-results'
    for category, _logs in categorized_logs.items():
        plt.figure()
        for model_key, model_data in _logs.items():
            model_data = model_data[['Epoch', 'AverageEpRet']]
            model_data.loc[:, 'Epoch'] = model_data['Epoch'].astype(int)
            sns.lineplot(data=model_data, x='Epoch', y='AverageEpRet', label=model_key)

        plt.title(category)
        plt.xlabel('Epoch')
        plt.ylabel('AverageEpRet')

        plt.savefig(f"{save_address}/{category}.png")
        plt.show()
        time.sleep(10)
    plt.close('all')

    print('[compare_scheduler_results.py] Plotting complete.')


def _plot_all_directories(algorithm: str = 'PPO', _get_model_folders: bool = False) -> None:
    """
    Plot results for the RLScheduler paper.
    """
    all_logs = []
    found_models = []
    folder_addresses = []
    root_dir = os.path.abspath(os.path.dirname(__file__) + '/logs/rl4sys-{}-scheduler'.format(str.lower(algorithm)))
    data = get_datasets(root_dir)

    for root, _, _ in os.walk(root_dir):
        if root != root_dir:
            folder_addresses.append(root)
    if folder_addresses:
        for folder in folder_addresses:
            progress_file_path = folder + '/progress.txt'
            if os.path.getsize(progress_file_path) > 0:
                try:
                    model_data = pd.read_table(progress_file_path)
                    if not model_data.empty:
                        found_models.append(model_data)
                except pd.errors.EmptyDataError:
                    print('[compare_scheduler_results.py] Skipping empty data file: {}'.format(progress_file_path))
            else:
                print('[compare_scheduler_results.py] Empty data file: {}'.format(progress_file_path))
        all_logs = found_models

    if not all_logs:
        print('[compare_scheduler_results.py] No models to plot...')
        return

    folder_names = {}
    for i, address in enumerate(folder_addresses):
        folder_names[i] = address.split('/')[-1]
    for i, data in enumerate(all_logs):
        data = data[['Epoch', 'AverageEpRet']]
        data.loc[:, 'Epoch'] = data['Epoch'].astype(int)
        plot = sns.lineplot(data=data, x='Epoch', y='AverageEpRet')
        plot.set_title(folder_names[i])
        plot.set_xlabel('Epoch')
        plot.set_ylabel('AverageEpRet')
        plot.figure.savefig(folder_addresses[i] + '/plot.png')
        plt.show()
        time.sleep(5)
    plt.close('all')
    print('[compare_scheduler_results.py] Plotting complete.')


def perform_scheduler_analysis(train_and_plot_models: bool = False, plot_all_directories: bool = False,
                               max_epochs: int = 100, algorithm: str = 'PPO') -> None:
    """
    Plot results for the RLScheduler paper.
    """
    if train_and_plot_models:
        _train_and_plot_models(max_epochs=max_epochs)

    if plot_all_directories:
        _plot_all_directories(algorithm=algorithm)


if __name__ == '__main__':
    """
        
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_and_plot_models', type=bool, default=False,
                        help='Train and plot models for scheduler comparison')
    parser.add_argument('--plot_all_directories', type=bool, default=False)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--algorithm', type=str, default='PPO')
    args, extras = parser.parse_known_args()

    perform_scheduler_analysis(train_and_plot_models=args.train_and_plot_models,
                               plot_all_directories=args.plot_all_directories, max_epochs=args.max_epochs)
