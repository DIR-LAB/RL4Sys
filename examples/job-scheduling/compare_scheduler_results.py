from utils.plot import *

import subprocess

"""
This script recreates the plots from the RLScheduler paper.
"""

""" Use Instructions:
----------------------------------------------------------------------------------------------------
First, train a model for each necessary performance metric. Use the following datasets:
    Lublin-1, HPC2N, SDSC-SP2, Lublin-2
Second, after training and logging progress.txt data, plot the results.
Third, compare results to RLScheduler paper: {https://arxiv.org/abs/1910.08925}.
"""


def train_scheduler_models() -> dict[str, str]:
    """
    Train all models for each performance metric and/or backfilling option from the RLScheduler paper.
    :return:
    """
    performance_metrics = ['bsld', 'wt', 'tt', 'ru', 'sld']
    datasets = ['data/lublin_1024.swf', 'data/HPC2N-2002-2.2-cln.swf', 'data/SDSC-SP2-1998-4.2-cln.swf',
                'data/lublin_256.swf']

    logs = {}
    iteration = 1
    while iteration <= 2:
        backfil = False if iteration == 1 else True
        for dataset in datasets:
            for metric in performance_metrics:
                # train models according to parameters
                data = _run_training_subprocess(dataset, performance_metrics.index(metric), backfil)

                key = '{}_{}_{}'.format(dataset, metric, 'backfil' if backfil else 'nobackfil')
                logs[key] = data
        iteration += 1

    return logs


def _run_training_subprocess(workload: str, job_score_type: int, backfil: bool, start_server: str = 'PPO',
                             num_iterations: int = 1, seed: int = 0):
    """
    Run a training subprocess with the specified parameters.
    """
    base_command = ['python', './scheduler.py', '--start-server={}'.format(start_server)]
    task_args = ['--workload={}'.format(workload), '--seed={}'.format(seed), '--job_score_type={}'.format(job_score_type),
                 '--backfil={}'.format(backfil), '--number-of-iterations={}'.format(num_iterations)]
    command = base_command + task_args
    file_dir = 'logs/rl4sys-{}-scheduler'.format(str.lower(start_server))
    data = None

    print('Running subprocess: [{}]'.format(' '.join(command)))
    try:
        subprocess.run(command, check=True, text=True, timeout=60)
        data = get_newest_dataset(file_dir)
    except subprocess.TimeoutExpired:
        print('Training subprocess timed out.')
    except subprocess.CalledProcessError as e:
        print(e)
    except Exception as e:
        print(e)

    return data


if __name__ == '__main__':
    model_logs = train_scheduler_models()

    for model_key, model_data in model_logs.items():
        print('Plotting data for model {}:'.format(model_key))
        plot_data(model_data)
