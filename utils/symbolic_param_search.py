from pysr import PySRRegressor
from skopt import Optimizer

import numpy as np
import torch

import os

from utils.logger import EpochLogger, setup_logger_kwargs
from utils.plot import load_training_data, data_normalization

import conf_loader
config = conf_loader.ConfigLoader()
load_model_path = config.load_model_path


class SymbolicRegressor:
    def __init__(self, niterations: int, binary_operators: list[str], unary_operators: list[str], X_initial, y_initial):
        self.regressor = PySRRegressor(
            niterations=niterations,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
        )

        self.X_initial = X_initial
        self.y_initial = y_initial

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)


class SymbolicOptimizer:
    def __init__(self, model_path, regression_model: SymbolicRegressor, hyperparam_dims):
        self.loaded_model = torch.load(model_path)
        self.regressor = regression_model
        self.optimizer = Optimizer(
            dimensions=[hyperparam_dims],
            base_estimator=regressor.predict,
        )

        current_dir = os.getcwd()
        log_data_dir = os.path.join(current_dir, './logs/')
        logger_kwargs = setup_logger_kwargs("rl4sys-symbolic-search", data_dir=log_data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        self.iteration = 0

    def log_iteration(self):
        self.logger.log_tabular('Iteration', self.iteration)
        self.logger.log_tabular('next_X', with_min_and_max=True)
        self.logger.log_tabular('Performance', with_min_and_max=True)
        self.logger.dump_tabular()

    def bayesian_optimization(self, n_calls: int):
        if n_calls > 0:
            for i in range(n_calls):
                next_x = self.optimizer.ask()
                performance = self.loaded_model(next_x)
                self.logger.store(next_X=next_x, Performance=performance)

                self.optimizer.tell(next_x, performance)
                self.iteration += 1
                self.log_iteration()
        else:
            print("[Symbolic Search] Number of optimization calls must be greater than 0")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='PPO',
                        help='Algorithm\'s results to use for symbolic regression')
    parser.add_argument('--model-path', type=str, default=load_model_path,
                        help='Path to the model to be used for symbolic regression')
    parser.add_argument('--data-folder', type=str, default='/Users/tybg/Documents/GitHub/RL4Sys/examples/job-scheduling/logs/rl4sys-ppo-scheduler/rl4sys-ppo-scheduler_s372120000',
                        help='Folder containing the training data files to be used for symbolic regression; please'
                             'provide the absolute path')
    parser.add_argument('--hyperparameters', type=str, default='gamma,lam,pi_lr,vf_lr',
                        help='Hyperparameters to optimize for in the symbolic regression')
    parser.add_argument('--performance-metrics', type=str, default='AverageEpRet,StdEpRet',
                        help='Performance metrics to optimize for in the symbolic regression')
    parser.add_argument('--save-model-path', type=str, default='model.pth',
                        help='Path to save the symbolic regression model')
    parser.add_argument('--niterations', type=int, default=100,
                        help='Number of iterations to run the symbolic regression for')
    parser.add_argument('--binary_operators', type=str, default='+,-,*,/',
                        help='List of binary operators to use in the symbolic regression')
    parser.add_argument('--unary_operators', type=str, default='sin,cos,exp',
                        help='List of unary operators to use in the symbolic regression')
    parser.add_argument('--optim-calls', type=int, default=200,
                        help='Number of calls to the Bayesian optimizer')
    parser.add_argument('--normalization', type=int, default=1,
                        help='Whether to normalize the data before training the symbolic regression model')
    args = parser.parse_args()

    ''' Below is a bunch of preprocessing and the eventual training execution process '''

    # turns operator strings into list collections
    binary_operators = args.binary_operators.split(',')
    unary_operators = args.unary_operators.split(',')

    # turns param and metric strings into list collections
    hyperparameters = args.hyperparameters.split(',')
    performance_metrics = args.performance_metrics.split(',')

    # loads all training data (dependent on chosen folder and algorithm)
    X_data, y_data = load_training_data(args.data_folder, args.algorithm)
    print(f"[Symbolic Search] Loaded training data for {args.algorithm} algorithm")

    # Stores the hyperparameters and performance metrics to be used in the symbolic regression
    X_hyperparams = [X_data[param] for param in hyperparameters]
    y_performance_metric = [y_data[metric] for metric in performance_metrics]

    # Basic hyperparameter bounding where min and max values are sourced from all collected hyperparam data
    current_param_bounds = [(min(X_hyperparams[i]), max(X_hyperparams[i])) for i in range(len(X_hyperparams))]
    if args.normalization > 0:
        y_data = data_normalization(y_data, args.normalization)
        print(f"[Symbolic Search] Normalized performance metrics")

    # Initialize and fit the symbolic regression model
    regressor = SymbolicRegressor(args.niterations, binary_operators, unary_operators, X_hyperparams, y_performance_metric)
    regressor.fit(X_hyperparams, y_performance_metric)
    print(f"[Symbolic Search] Fitted symbolic regression model with {args.niterations} iterations")

    # Initialize and optimize the symbolic regression model using Bayesian optimization
    bayesian_optimizer = SymbolicOptimizer(args.model_path, regressor, current_param_bounds)
    print(f"[Symbolic Search] Initialized Bayesian optimizer with {args.optim_calls} optimization calls")
    bayesian_optimizer.bayesian_optimization(args.optim_calls)

    # Output and save the best hyperparameters found by the optimizer
    best_hyperparams = bayesian_optimizer.optimizer.Xi[np.argmin(bayesian_optimizer.optimizer.yi)]
    print(f"[Symbolic Search] Best hyperparameters: {best_hyperparams}")
    with open('symbolic_search_best_hyperparams.txt', 'w') as f:
        f.write(str(best_hyperparams))
