import os

import numpy as np
from pysr import PySRRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from skopt import Optimizer
from skopt.space import Integer, Real

import torch

from collections import defaultdict

import conf_loader
from utils.logger import EpochLogger, setup_logger_kwargs
from utils.plot import load_training_data, data_normalization

config = conf_loader.ConfigLoader()
load_model_path = config.load_model_path

# Below are all the available hyperparameters for the symbolic search.
# This can be altered to fit for new algorithms. Cite config.json for any possibly missing hyperparameters.
# Generally, however, we want to keep this list as comprehensive as possible.
# This is certainly a maintainability issue....
COMMON_HYPERPARAM_BOUNDS = {
    "Integer": {
        "batch_size": (16, 512),             # A typical range for batch sizes
        "seed": (0, 100),                    # Seed values generally range from 0 to some arbitrary number
        "traj_per_epoch": (1, 20),           # Number of trajectories per epoch
        "train_update_freq": (1, 20),        # Frequency of updates
        "train_q_iters": (10, 200),          # Iterations for Q-learning updates
        "train_pi_iters": (10, 100),         # Number of policy training iterations
        "train_v_iters": (10, 100),          # Number of value function training iterations
        "train_iters": (10, 500),            # Training iterations
        "log_std_min": (-20, -1),            # Minimum log standard deviation
        "log_std_max": (1, 2),               # Maximum log standard deviation
    },
    "Real": {
        "gamma": (0.90, 0.999),              # Discount factor range commonly used in RL
        "epsilon": (0.01, 1.0),              # Exploration rate for epsilon-greedy strategy
        "epsilon_min": (0.01, 0.1),          # Minimum epsilon value
        "epsilon_decay": (1e-5, 1e-2),       # Decay rate for epsilon
        "q_lr": (1e-5, 1e-3),                # Learning rate for Q-networks
        "clip_ratio": (0.05, 0.3),           # Clip ratio
        "lam": (0.9, 0.999),                 # Lambda for GAE (Generalized Advantage Estimation)
        "pi_lr": (1e-5, 1e-3),               # Learning rate for policy (pi)
        "vf_lr": (1e-5, 1e-3),               # Learning rate for value function (vf)
        "target_kl": (0.01, 0.05),           # Target KL divergence
        "polyak": (0.001, 0.1),              # Polyak averaging factor for target networks
        "alpha": (1e-4, 0.5),                # Entropy coefficient (alpha)
        "lr": (1e-5, 1e-3),                  # Learning rate
        "clip_grad_norm": (0.1, 5),          # Gradient clipping threshold
    }
}


def _log_iteration(iter_logger: EpochLogger, iteration: int):
    iter_logger.log_tabular('Iteration', iteration)
    iter_logger.log_tabular('next_X')
    iter_logger.log_tabular('Performance', with_min_and_max=True)
    iter_logger.dump_tabular()


class SymbolicRegressor:
    def __init__(self, niterations: int, binary_operators: list[str], unary_operators: list[str], X_initial, y_initial):
        self.regressor = PySRRegressor(
            niterations=niterations,
            ncycles_per_iteration=10,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
        )

        self.X_initial = X_initial
        self.y_initial = y_initial
        self.mean_hat = None
        self.std_hat = None

    def fit(self, X, y):
        self.regressor.fit(X, y)

    def predict(self, X, return_std=False):
        if return_std:
            return self.mean_hat, self.std_hat
        return self.regressor.predict(X)


class SymbolicOptimizer:
    def __init__(self, model_path, regression_model: SymbolicRegressor, hyperparam_dims):
        self.loaded_model = torch.load(model_path)
        self.regressor = regression_model
        dimension_list = [
                 Integer(low=COMMON_HYPERPARAM_BOUNDS['Integer'][param][0], high=COMMON_HYPERPARAM_BOUNDS['Integer'][param][1])
                 for param in hyperparam_dims if param in COMMON_HYPERPARAM_BOUNDS['Integer']
             ] + [
                 Real(low=COMMON_HYPERPARAM_BOUNDS['Real'][param][0], high=COMMON_HYPERPARAM_BOUNDS['Real'][param][1])
                 for param in hyperparam_dims if param in COMMON_HYPERPARAM_BOUNDS['Real']
             ]
        self.optimizer = Optimizer(
            dimensions=dimension_list,
            base_estimator=self.regressor.regressor,
            acq_func_kwargs={'return_std': False}
        )

        self.iteration = 0

    def bayesian_optimization(self, n_calls: int, logger: EpochLogger):
        if n_calls > 0:
            for i in range(n_calls):
                next_x = self.optimizer.ask()
                performance = self.regressor.predict(np.array(next_x).reshape(1, -1))

                # Ensure performance is a scalar
                if performance.ndim > 1:
                    performance = performance.flatten()  # Convert to 1D array
                if len(performance) > 1:
                    performance = performance[0]  # Take the first element if it's an array

                # y_opt = min(logger.get_stats('Performance'))
                #
                # mean, std = self.regressor.predict(np.array(next_x).reshape(1, -1), return_std=True)
                #
                # logger.store(next_X=next_x, Performance=performance)

                self.optimizer.tell(next_x, performance)
                self.iteration += 1
                # _log_iteration(logger, self.iteration)
        else:
            print("[Symbolic Search] Number of optimization calls must be greater than 0")


if __name__ == '__main__':
    """Symbolic Hyperparameter Search using PySR and Bayesian Optimization
    
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='PPO',
                        help='Algorithm\'s results to use for symbolic regression')
    parser.add_argument('--model-path', type=str, default=load_model_path,
                        help='Path to the model to be used for symbolic regression')
    parser.add_argument('--data-folder', type=str, default='/Users/tybg/Documents/GitHub/RL4Sys/examples/job-scheduling/logs/rl4sys-ppo-scheduler/',
                        help='Folder containing the training data files to be used for symbolic regression; please'
                             'provide the absolute path')
    parser.add_argument('--hyperparameters', type=str, default='traj_per_epoch,clip_ratio,gamma,lam,pi_lr,'
                                                               'vf_lr,train_pi_iters,train_v_iters,target_kl',
                        help='Hyperparameters to optimize for in the symbolic regression')
    parser.add_argument('--performance-metrics', type=str, default='MaxEpRet,MinEpRet',
                        help='Performance metrics to optimize for in the symbolic regression')
    parser.add_argument('--save-model-path', type=str, default='model.pth',
                        help='Path to save the symbolic regression model')
    parser.add_argument('--niterations', type=int, default=3,
                        help='Number of iterations to run the symbolic regression for')
    parser.add_argument('--binary_operators', type=str, default='+,-,*,/',
                        help='List of binary operators to use in the symbolic regression')
    parser.add_argument('--unary_operators', type=str, default='sin,cos,exp,abs',
                        help='List of unary operators to use in the symbolic regression')
    parser.add_argument('--optim-calls', type=int, default=200,
                        help='Number of calls to the Bayesian optimizer')
    parser.add_argument('--normalization', type=int, default=1,
                        help='Whether to normalize the data before training the symbolic regression model')
    args = parser.parse_args()

    current_dir = os.getcwd()
    log_data_dir = os.path.join(current_dir, './logs/')
    logger_kwargs = setup_logger_kwargs("rl4sys-symbolic-search-" + args.algorithm, data_dir=log_data_dir)
    logger = EpochLogger(**logger_kwargs)
    config_data = {k: v for k, v in locals().items() if isinstance(v, (dict, list, str, int, float))}
    logger.save_config(config_data)

    ''' Below is a bunch of preprocessing and the eventual training execution process '''
    print(f"[Symbolic Search] Starting symbolic search for {args.algorithm} algorithm")

    # turns operator strings into list collections
    print(f"[Symbolic Search] Using the following operators: {args.binary_operators}, {args.unary_operators}")
    binary_operators = args.binary_operators.split(',')
    unary_operators = args.unary_operators.split(',')

    # turns param and metric strings into list collections
    print(f"[Symbolic Search] Using the following hyperparameters: {args.hyperparameters}")
    hyperparameters = args.hyperparameters.split(',')
    performance_metrics = args.performance_metrics.split(',')

    # Basic hyperparameter bounding where min and max values are sourced from all collected hyperparam data
    print(f"[Symbolic Search] Setting hyperparameter bounds")
    current_param_bounds = {param: COMMON_HYPERPARAM_BOUNDS['Real'][param] if param in COMMON_HYPERPARAM_BOUNDS['Real']
                            else COMMON_HYPERPARAM_BOUNDS['Integer'][param] for param in hyperparameters}

    # loads all training data (dependent on chosen folder and algorithm)
    print(f"[Symbolic Search] Loading training data for {args.algorithm} algorithm")
    X_data, y_data = load_training_data(args.data_folder, args.algorithm)

    # aggregates all data into single dictionary if multi-folder training data
    if isinstance(X_data, list) and isinstance(y_data, list):
        if all(isinstance(value, dict) for value in X_data) and all(isinstance(value, dict) for value in y_data):
            print(f"[Symbolic Search] Aggregating training data from multiple folders")
            merged_X = defaultdict(list)
            merged_y = defaultdict(list)
            for nested_X in X_data:
                for key, value in nested_X.items():
                    merged_X[key].append(value)
            for nested_y in y_data:
                for key, value in nested_y.items():
                    merged_y[key].append(value)
            X_data = dict(merged_X)
            y_data = dict(merged_y)

    # Stores the hyperparameters and performance metrics to be used in the symbolic regression
    print(f"[Symbolic Search] Extracting hyperparameters and performance metrics")
    X_hyperparams = {param: X_data[param] for param in hyperparameters}
    y_performance_metrics = {metric: np.concatenate(y_data[metric]) for metric in performance_metrics}

    X_hyperparams_np = np.column_stack([X_hyperparams[feature] for feature in hyperparameters])
    y_performance_metrics_np = np.column_stack([y_performance_metrics[metric] for metric in performance_metrics]).T

    # Normalize the data if specified
    if args.normalization > 0:
        print(f"[Symbolic Search] Normalizing hyperparams & performance metrics")
        X_hyperparams_np = data_normalization(X_hyperparams_np, args.normalization)
        y_performance_metrics_np = data_normalization(y_performance_metrics_np, args.normalization)

    X_test, X_train, y_test, y_train = train_test_split(X_hyperparams_np, y_performance_metrics_np, test_size=0.2,
                                                        train_size=0.8, random_state=42, shuffle=False)

    # Initialize and fit the symbolic regression model
    print(f"[Symbolic Search - Training] Initializing PySR symbolic regression model")
    regressor = SymbolicRegressor(args.niterations, binary_operators, unary_operators, X_train, y_train)
    regressor.fit(X_train, y_train)
    print(f"[Symbolic Search - Training] Fitted symbolic regression model with {args.niterations} iterations")

    y_test_hat = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_test_hat)
    print(f"[Symbolic Search - Training] Mean Squared Error: {mse}")

    # Initialize and optimize the symbolic regression model using Bayesian optimization
    print(f"[Symbolic Search - Optimization] Initializing Bayesian optimizer with {args.optim_calls} optimization calls")
    bayesian_optimizer = SymbolicOptimizer(args.model_path, regressor, current_param_bounds)
    bayesian_optimizer.bayesian_optimization(args.optim_calls, logger)
    print(f"[Symbolic Search - Optimization] Completed Bayesian optimization")

    # Output and save the best hyperparameters found by the optimizer
    best_hyperparams = bayesian_optimizer.optimizer.Xi[np.argmin(bayesian_optimizer.optimizer.yi)]
    print(f"[Symbolic Search] Best hyperparameters: {best_hyperparams}")
    with open('symbolic_search_best_hyperparams.txt', 'w') as f:
        f.write(str(best_hyperparams))
