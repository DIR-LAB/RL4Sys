from pysr import PySRRegressor
from skopt import Optimizer
from skopt.acquisition import gaussian_ei
import torch

import numpy as np
import math

import sys, os

from utils.plot import load_training_data, data_normalization

import conf_loader
config = conf_loader.ConfigLoader()
load_model_path = config.load_model_path


class SymbolicRegressor:
    def __init__(self, niterations: int, binary_operators: list[str], unary_operators: list[str], populations: int,
                 population_size: int):
        self.regressor = PySRRegressor(
            niterations=niterations,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            populations=populations,
            population_size=population_size
        )

        self.X_hyperparams = X_hyperparams
        self.y_performance_metric = y_performance_metric

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X):
        return self.regressor.predict(X)


class SymbolicOptimizer:
    def __init__(self, model_path, regressor: SymbolicRegressor, hyperparam_dims):
        self.loaded_model = torch.load(model_path)
        self.regressor = regressor
        self.optimizer = Optimizer(
            dimensions=[hyperparam_dims],
            base_estimator=regressor.predict,
        )

    def log_epoch(self):
        self.logger.log_tabular('Iteration', self.iteration)
        self.logger.log_tabular('Performance', with_min_and_max=True)
        self.logger.dump_tabular()

    def bayesian_optimization(self, n_calls: int):
        if n_calls > 0:
            for i in range(n_calls):
                next_x = self.optimizer.ask()
                performance = self.loaded_model(next_x)

                self.optimizer.tell(next_x, performance)
                if i % math.floor(n_calls / 10) == 0:
                    log_epoch()
        else:
            print("[Symbolic Search] Number of optimization calls must be greater than 0")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='PPO',
                        help='Algorithm\'s results to use for symbolic regression')
    parser.add_argument('--model-path', type=str,
                        help='Path to the model to be used for symbolic regression')
    parser.add_argument('--data-folder', type=str,
                        help='Folder containing the training data files to be used for symbolic regression')
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
    parser.add_argument('--populations', type=int, default=15,
                        help='Number of populations to use in the symbolic regression')
    parser.add_argument('--population-size', type=int, default=100,
                        help='Number of individuals in each population')
    parser.add_argument('--optim-calls', type=int, default=200,
                        help='Number of calls to the Bayesian optimizer')
    parser.add_argument('--normalization', type=int, default=1,
                        help='Whether to normalize the data before training the symbolic regression model')
    args = parser.parse_args()

    binary_operators = args.binary_operators.split(',')
    unary_operators = args.unary_operators.split(',')

    X_data, y_data = load_training_data(args.data_folder, args.algorithm)
    print(f"[Symbolic Search] Loaded training data for {args.algorithm} algorithm")

    X_hyperparams = [X_data[param] for param in args.hyperparameters.split(',')]
    y_performance_metric = [y_data[metric] for metric in args.performance_metrics.split(',')]

    current_param_bounds = [(min(X_hyperparams[i]), max(X_hyperparams[i])) for i in range(len(X_hyperparams))]
    if args.normalization > 0:
        y_data = data_normalization(y_data, args.normalization)
        print(f"[Symbolic Search] Normalized performance metrics")

    regressor = SymbolicRegressor(args.niterations, binary_operators, unary_operators, args.populations,
                                  args.population_size)
    regressor.fit(X_hyperparams, y_performance_metric)
    print(f"[Symbolic Search] Fitted symbolic regression model with {args.niterations} iterations")

    bayesian_optimizer = SymbolicOptimizer(args.model_path, regressor, current_param_bounds)
    print(f"[Symbolic Search] Initialized Bayesian optimizer with {args.optim_calls} optimization calls")
    bayesian_optimizer.bayesian_optimization(args.optim_calls)

    best_hyperparams = bayesian_optimizer.optimizer.Xi[np.argmin(bayesian_optimizer.optimizer.yi)]
    print(f"[Symbolic Search] Best hyperparameters: {best_hyperparams}")
