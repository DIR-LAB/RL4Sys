import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from _common._examples.BaseApplication import ApplicationAbstract

import numpy as np
import random
import time

import math
import torch

import pygame

from agent import RL4SysAgent
from training_server import TrainingServer

from utils.plot import get_newest_dataset

"""
Environment script: Maze Game Simulator

Training server parameters:
    kernel_size | MAX_SIZE = 5
    kernel_dim  | FEATURES = 4
    buf_size    | MOVE_SEQUENCE_SIZE * 100 = 500000
"""

GAME_ELEMENTS = {'path': 0, 'wall': 1, 'pitfall': 2, 'start': 3, 'goal': 4}


def write_maze_to_log_dir(maze: np.ndarray, log_dir: str):
    np.save(f"{log_dir}/maze_{time.time()}", maze)


def read_maze_from_file(file_address: str):
    try:
        maze = np.load(file_address)
    except FileNotFoundError:
        print(f"[maze.py] Maze file not found at {file_address}")
        return None
    return maze


class MazeGenerator:
    def __init__(self, area_dimensions: tuple[int, int] = (10, 10), start_position: tuple[int, int] = (0, 0),
                 goal_position: tuple[int, int] = (9, 9), pitfall_prob: float = 0.1, maze: np.ndarray = None):
        if maze is not None:
            self._maze, self._area_dimensions, self._start, self._goal, self._pitfall_prob = self._maze_element_search(
                maze)
            assert self._area_dimensions[0] > 0 and self._area_dimensions[1] > 0
            assert self._start is not None
            assert self._goal is not None
        else:
            self._maze = np.ones(area_dimensions, dtype=int)  # sets maze to all walls
            self._area_dimensions = area_dimensions
            self._start = start_position
            self._goal = goal_position
            self._pitfall_prob = pitfall_prob
            self._carve_paths()
            print(f"[maze.py - MazeGenerator] Paths carved")
            self._place_pitfalls()
            print(f"[maze.py - MazeGenerator] Pitfalls placed") if pitfall_prob > 0 else (
                print('[maze.py - MazeGenerator] No pitfalls placed'))
            self._mark_positions()
            print(f"[maze.py - MazeGenerator] Maze environment generation complete")

    def _maze_element_search(self, maze: np.ndarray):
        area_dimensions = (len(maze), len(maze[0]))
        start = None
        goal = None
        pitfall_prob = 0
        for i in range(area_dimensions[0]):
            for j in range(area_dimensions[1]):
                if maze[i, j] == GAME_ELEMENTS['start']:
                    start = (i, j)
                elif maze[i, j] == GAME_ELEMENTS['goal']:
                    goal = (i, j)
                elif maze[i, j] == GAME_ELEMENTS['pitfall']:
                    pitfall_prob += 1

        pitfall_prob = pitfall_prob / (area_dimensions[0] * area_dimensions[1])
        return maze, area_dimensions, start, goal, pitfall_prob

    def _carve_paths(self):
        def _is_within_bounds(x, y):
            return 0 <= x < self._area_dimensions[0] and 0 <= y < self._area_dimensions[1]

        def _carve(x, y):
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx * 2, y + dy * 2
                if _is_within_bounds(nx, ny) and self._maze[nx, ny] == GAME_ELEMENTS['wall']:
                    self._maze[x + dx, y + dy] = GAME_ELEMENTS['path']
                    self._maze[nx, ny] = GAME_ELEMENTS['path']
                    _carve(nx, ny)

        x, y = self._start
        self._maze[x, y] = GAME_ELEMENTS['path']
        _carve(x, y)

    def _place_pitfalls(self):
        available_area = [(i, j) for i in range(self._area_dimensions[0]) for j in range(self._area_dimensions[1]) if
                          self._maze[i, j] == GAME_ELEMENTS['wall']]
        num_pitfalls = int(len(available_area) * self._pitfall_prob)
        pitfalls = random.sample(available_area, num_pitfalls)
        for (i, j) in pitfalls:
            self._maze[i, j] = GAME_ELEMENTS['pitfall']

    def _mark_positions(self):
        self._maze[self._start] = GAME_ELEMENTS['start']
        self._maze[self._goal] = GAME_ELEMENTS['goal']

        # Ensure the goal position has at least one adjacent path
        goal_x, goal_y = self._goal
        adjacent_positions = [(goal_x - 1, goal_y), (goal_x + 1, goal_y), (goal_x, goal_y - 1), (goal_x, goal_y + 1)]
        for x, y in adjacent_positions:
            if 0 <= x < self._area_dimensions[0] and 0 <= y < self._area_dimensions[1]:
                if self._maze[x, y] == GAME_ELEMENTS['path']:
                    return  # At least one adjacent path exists, no need to carve
        # If no adjacent path exists, carve one
        for x, y in adjacent_positions:
            if 0 <= x < self._area_dimensions[0] and 0 <= y < self._area_dimensions[1]:
                self._maze[x, y] = GAME_ELEMENTS['path']
                break

    def get(self):
        return self._maze, self._area_dimensions, self._start, self._goal, self._pitfall_prob


class AgentProperties:
    def __init__(self, maze: np.ndarray, start_position: tuple[int, int], goal_position: tuple[int, int],
                 play_new_levels: bool = False):
        self.maze = maze
        self.start = start_position
        self.position = start_position
        self.goal = goal_position
        self.play_new_levels = play_new_levels

        self.GOAL_REWARD = 1000 if not self.play_new_levels else 500

    def move(self, direction):
        x, y = self.position
        if direction == 'up' or direction == 0:
            self.position = (x - 1, y)
        elif direction == 'down' or direction == 1:
            self.position = (x + 1, y)
        elif direction == 'left' or direction == 2:
            self.position = (x, y - 1)
        elif direction == 'right' or direction == 3:
            self.position = (x, y + 1)
        return self.position

    def calculate_reward(self, previous_position: tuple[int, int], next_position: tuple[int, int]):
        px, py = previous_position
        nx, ny = next_position

        def granular_move_reward():
            def towards_goal() -> bool | None:
                if previous_position == next_position:
                    return None
                goal_x, goal_y = self.goal
                start_x, start_y = self.start
                # Check if the new position is closer to the goal than the previous position
                # and further from the start than the previous position
                return (abs(nx - goal_x) + abs(ny - goal_y) < abs(px - goal_x) + abs(py - goal_y)) and \
                    (abs(nx - start_x) + abs(ny - start_y) > abs(px - start_x) + abs(py - start_y))

            if towards_goal():
                return .01
            else:
                return -.05

        def element_reward():
            if not (0 <= nx < self.maze.shape[0] and 0 <= ny < self.maze.shape[1]):
                return -1.0
            elif self.maze[nx, ny] == GAME_ELEMENTS['wall']:
                return -1.0
            elif self.maze[nx, ny] == GAME_ELEMENTS['pitfall']:
                return -2.0
            elif self.maze[nx, ny] == GAME_ELEMENTS['goal']:
                return self.GOAL_REWARD
            else:
                return 0.1

        return granular_move_reward() + element_reward()

    def draw_agent(self, screen):
        x, y = self.position
        color = (0, 128, 128)  # Teal
        pygame.draw.circle(screen, color, (x * 20 + 10, y * 20 + 10), 8.5)


FEATURES = 4
MAX_SIZE = 5

MOVE_SEQUENCE_SIZE = 500


class MazeGameSim(ApplicationAbstract):
    def __init__(self, seed, model=None, maze: MazeGenerator = None, area_dimensions: tuple[int, int] = (6, 6),
                 static_area_dimensions: bool = False, play_new_levels: int = 0, enable_pitfalls: bool = False,
                 performance_metric: int = 0, render_game: bool = False):
        super().__init__()
        self._area_dimensions = area_dimensions
        self._seed = seed
        self._play_new_levels = play_new_levels
        self._enable_pitfalls = enable_pitfalls
        self._static_area_dimensions = static_area_dimensions
        self._performance_metric = performance_metric

        self._render_game = render_game

        self._MAX_NEW_MAZE_SIZE = 20

        self._relative_goal_location = None

        self._level_counter = 0

        if seed < 0:
            print(f"[maze.py] Seed value must be a positive integer, setting to 0")
            seed = 0

        seed_seq = np.random.SeedSequence(seed)
        self.np_random = np.random.Generator(np.random.PCG64(seed_seq))

        if maze is None:
            self.set_new_maze()
        else:
            self.maze, self._area_dimensions, self._start_position, self._goal_position, self._pitfall_prob = maze.get()

        if self._render_game:
            self._screen = pygame.display.set_mode((self.maze.shape[0] * 20, self.maze.shape[1] * 20))
            pygame.display.set_caption('Maze Game Simulator')

        self.rlagent = RL4SysAgent(model=model)
        self.agent_properties = None

        self.simulator_stats = {'moves': 0, 'action_rewards': [], 'performance_rewards': [], 'success_count': 0,
                                'death_count': 0, 'collision_count': 0, 'time_to_goal': [], 'time_to_death': [],
                                'total_iterations': 0}

    def set_new_maze(self):
        length_x, length_y = self._area_dimensions
        start_goal_min_distance = max(length_x, length_y) // 2 + min(length_x, length_y) // 2
        self._start_position = (random.randint(0, length_x - 1), random.randint(0, length_y - 1))
        self._goal_position = (random.randint(0, length_x - 1), random.randint(0, length_y - 1))

        while True:
            self._goal_position = (random.randint(0, length_x - 1), random.randint(0, length_y - 1))
            if (abs(self._goal_position[0] - self._start_position[0]) +
                abs(self._goal_position[1] - self._start_position[1])) >= start_goal_min_distance:
                break

        GOAL_POSITIONS = {'above': 1, 'below': 2, 'left': 3, 'right': 4}

        valid_goal_positions = []
        for x in range(length_x):
            for y in range(length_y):
                if x != self._start_position[0] and y != self._start_position[1]:
                    distance = math.sqrt((x - self._start_position[0]) ** 2 + (y - self._start_position[1]) ** 2)
                    if distance >= start_goal_min_distance:
                        relative_location = GOAL_POSITIONS['above'] if y < self._start_position[1] else (
                            GOAL_POSITIONS['below'] if y > self._start_position[1] else (
                                GOAL_POSITIONS['left'] if x < self._start_position[0] else
                                GOAL_POSITIONS['right']))
                        if relative_location != self._relative_goal_location:
                            valid_goal_positions.append((x, y, relative_location))
        if valid_goal_positions:
            chosen_position = random.choice(valid_goal_positions)
            self._goal_position, self._relative_goal_location = (chosen_position[0], chosen_position[1]), \
                chosen_position[2]

        self._pitfall_prob = random.uniform(0.1, 0.333) if self._enable_pitfalls else 0
        self.maze, _, _, _, _ = MazeGenerator(self._area_dimensions, self._start_position, self._goal_position,
                                              self._pitfall_prob).get()

        if self._render_game:
            self._screen = pygame.display.set_mode((self.maze.shape[0] * 20, self.maze.shape[1] * 20))

    def draw_sim(self):
        self._screen.fill((0, 0, 0))

        for x in range(self.maze.shape[0]):
            for y in range(self.maze.shape[1]):
                color = ()
                if self.maze[x, y] == GAME_ELEMENTS['path']:
                    color = (255, 255, 255)  # White
                elif self.maze[x, y] == GAME_ELEMENTS['wall']:
                    color = (0, 0, 0)  # Black
                elif self.maze[x, y] == GAME_ELEMENTS['pitfall']:
                    color = (255, 0, 0)  # Red
                elif self.maze[x, y] == GAME_ELEMENTS['start']:
                    color = (255, 255, 0)  # Yellow
                elif self.maze[x, y] == GAME_ELEMENTS['goal']:
                    color = (0, 255, 0)  # Green
                pygame.draw.rect(self._screen, color, (x * 20, y * 20, 20, 20))

        self.agent_properties.draw_agent(self._screen)
        pygame.display.flip()

    def run_application(self, num_iterations: int = 1, num_moves: int = 500, play_new_levels: bool = False):
        def reset_agent():
            self.agent_properties.position = self._start_position

        def check_win(position):
            return position == self._goal_position

        def check_death(position):
            x, y = position
            if 0 <= x < self.maze.shape[0] and 0 <= y < self.maze.shape[1]:
                return self.maze[x, y] == GAME_ELEMENTS['pitfall']
            return False

        def check_move_validity(position):
            x, y = position
            return 0 <= x < self.maze.shape[0] and 0 <= y < self.maze.shape[1]

        # simulator main loop
        self.agent_properties = AgentProperties(self.maze, self._start_position, self._goal_position, play_new_levels)
        self._level_counter += 1

        maze_logged = False
        clock = pygame.time.Clock()
        clock.tick(144)
        for iteration in range(num_iterations):
            self.simulator_stats['total_iterations'] += 1
            reset_agent()
            moves, rl_runs = 0, 0
            print(f"[maze.py - simulator] Game Iteration {iteration}")
            while moves < num_moves:
                if not maze_logged and moves / (MOVE_SEQUENCE_SIZE / 4) >= 1:
                    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'logs'))
                    newest_log_dir = get_newest_dataset(log_dir, return_file_root=True)
                    write_maze_to_log_dir(self.maze, newest_log_dir)
                    maze_logged = True

                # Render game
                if self._render_game:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                    self.draw_sim()

                # Make initial move
                start_time = clock.get_time()
                current_position = self.agent_properties.position
                rew = self.agent_properties.calculate_reward(current_position, current_position)
                self.simulator_stats['action_rewards'].append(rew)
                obs, mask = self.build_observation(current_position)
                rl4sys_action = self.rlagent.request_for_action(obs, mask, rew)
                rl_runs += 1
                moves += 1
                self.simulator_stats['moves'] += 1

                while True:
                    if moves >= num_moves:
                        break

                    # Validate move and new position
                    # If invalid, reset agent position to previous position
                    previous_position = self.agent_properties.position
                    new_position = self.agent_properties.move(rl4sys_action.act[0]
                                                              if isinstance(rl4sys_action.act, np.ndarray)
                                                              else rl4sys_action.act)
                    new_rew = self.agent_properties.calculate_reward(previous_position, new_position)
                    self.simulator_stats['action_rewards'].append(new_rew)
                    obs, mask = self.build_observation(new_position)
                    rl4sys_action = self.rlagent.request_for_action(obs, mask, new_rew)
                    rl_runs += 1
                    moves += 1
                    self.simulator_stats['moves'] += 1

                    if not check_move_validity(new_position):
                        self.simulator_stats['collision_count'] += 1
                        self.agent_properties.position = previous_position
                    elif self.maze[new_position] == GAME_ELEMENTS['wall']:
                        self.simulator_stats['collision_count'] += 1
                        self.agent_properties.position = previous_position
                    elif self.maze[new_position] == GAME_ELEMENTS['pitfall']:
                        self.simulator_stats['death_count'] += 1
                    elif self.maze[new_position] == GAME_ELEMENTS['goal']:
                        self.simulator_stats['success_count'] += 1
                        self.simulator_stats['time_to_goal'].append(clock.get_time() - start_time)

                    # flag last action
                    if rl_runs >= MOVE_SEQUENCE_SIZE:
                        print(f'[maze.py - simulator] RL4SysAgent moves made: {moves}')
                        self.simulator_stats['moves'] = moves
                        rl_runs = 0
                        rl_total = self.calculate_performance_return(self.simulator_stats)
                        rew = -rl_total
                        self.simulator_stats['performance_rewards'].append(rew)
                        self.rlagent.flag_last_action(rew)
                    if self._render_game:
                        self.draw_sim()
                    reset = check_win(new_position) or check_death(new_position)
                    if reset:
                        reset_agent()
                        break
        if play_new_levels:
            self.play_new_level(num_iterations, num_moves)

    def play_new_level(self, num_iterations: int = 1, num_moves: int = 500):
        """
        Play new level after each set of n iterations (if enabled)
        :param num_iterations: Number of iterations of n steps per level
        :param num_moves: Maximum number of moves allowed per iteration
        :return: new simulation instance
        """
        if not self._static_area_dimensions:
            x, y = self._area_dimensions
            nx, ny = x + self._level_counter, y + self._level_counter
            if nx > self._MAX_NEW_MAZE_SIZE or ny > self._MAX_NEW_MAZE_SIZE:
                nx, ny = 6, 6
                self._level_counter = 0
            self._area_dimensions = (nx, ny)
            self.set_new_maze()
            print(f"[maze.py] New level generated: ({nx}, {ny}) -> {self._area_dimensions}, loading...")
            self.run_application(num_iterations, num_moves, play_new_levels=True)
        else:
            print(f"[maze.py] Static area dimensions enabled, loading...")
            self.set_new_maze()
            self.run_application(num_iterations, play_new_levels=True)

    def build_observation(self, agent_position: tuple[int, int]):
        """
        Build observation vector for RL4Sys agent
        Build mask for RL4Sys agent

        Observation vector is a combined local and global view of the maze environment
        :return: observation vector, mask
        """
        # Make combined local-global observation vector :^)

        # Local observation construction
        # Important variables for calculations
        local_view_size = 3
        agent_x, agent_y = agent_position
        maze_size_x, maze_size_y = self.maze.shape

        # Determine local observation boundaries
        half_view_size = local_view_size // 2
        min_x = max(agent_x - half_view_size, 0)
        max_x = min(agent_x + half_view_size + 1, maze_size_x)
        min_y = max(agent_y - half_view_size, 0)
        max_y = min(agent_y + half_view_size + 1, maze_size_y)

        # Determine local observation coordinates
        local_obs_x_start = half_view_size - (agent_x - min_x)
        local_obs_x_end = local_obs_x_start + (max_x - min_x)
        local_obs_y_start = half_view_size - (agent_y - min_y)
        local_obs_y_end = local_obs_y_start + (max_y - min_y)

        # Initialize local observation with out-of-bound values
        local_obs = -np.ones((local_view_size, local_view_size))

        # Fill local observation with maze elements within boundaries
        local_obs[local_obs_x_start:local_obs_x_end, local_obs_y_start:local_obs_y_end] \
            = self.maze[min_x:max_x, min_y:max_y]

        # Agent's local position within local grid
        agent_view_x = half_view_size
        agent_view_y = half_view_size
        local_obs[agent_view_x, agent_view_y] = 9

        # flatten for RL4Sys agent, training, and vector combination
        flat_local_obs = local_obs.flatten()

        # Global observation construction
        # 11 for agent position, goal position, and additional features
        global_obs = np.zeros(11, dtype=float)

        # Agent's position in context to whole maze environment
        global_obs[0:2] = [agent_x, agent_y]

        # Goal position in context to whole maze environment
        goal_x, goal_y = self._goal_position
        global_obs[2:4] = [goal_x, goal_y]

        # Surrounding area elements
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        nearby_pitfall_count, nearby_wall_count = 0, 0
        for i, (dx, dy) in enumerate(directions):
            nx, ny = agent_x + dx, agent_y + dy
            if 0 <= nx < self.maze.shape[0] and 0 <= ny < self.maze.shape[1]:
                global_obs[4 + i] = self.maze[nx, ny]
                if self.maze[nx, ny] == GAME_ELEMENTS['pitfall']:
                    nearby_pitfall_count += 1
                elif self.maze[nx, ny] == GAME_ELEMENTS['wall']:
                    nearby_wall_count += 1
            else:
                global_obs[4 + i] = -1

        # Probability of pitfall in nearby area
        nearby_pitfall_prob = nearby_pitfall_count / 4
        global_obs[8] = nearby_pitfall_prob

        # Probability of wall in nearby area
        nearby_wall_prob = nearby_wall_count / 4
        global_obs[9] = nearby_wall_prob

        # Calculate distance to goal from agent
        manhattan_distance = -(abs(agent_x - goal_x) + abs(agent_y - goal_y))
        global_obs[10] = manhattan_distance

        # Combine flat local observation with global observation vector
        combined_observation = np.concatenate((flat_local_obs, global_obs))
        obs = torch.as_tensor(combined_observation, dtype=torch.float32)

        # Make mask :^)
        mask = np.zeros(MAX_SIZE, dtype=float)
        mask = torch.as_tensor(mask, dtype=torch.float32)
        return obs, mask

    def calculate_performance_return(self, elements) -> float:
        """
        Calculate performance score based on performance metric using captured simulator elements
        :return: returns calculated performance score
        """
        if self._performance_metric == 0:
            # avg cumulative reward per reward count
            return float(sum(elements['action_rewards']) / len(elements['action_rewards']))
        elif self._performance_metric == 1:
            # avg cumulative reward per success rate
            return float(elements['action_rewards'] / len(elements['success_count']))
        elif self._performance_metric == 2:
            # avg cumulative reward per death rate
            return -float(elements['action_rewards'] / len(elements['death_count']))
        elif self._performance_metric == 3:
            # avg cumulative reward per collision rate
            return -float(elements['action_rewards'] / len(elements['collision_count']))
        elif self._performance_metric == 4:
            # avg cumulative reward per failure rate
            return float(sum(elements['action_rewards']) /
                         (elements['collision_count'] + elements['death_count']))
        elif self._performance_metric == 5:
            # avg cumulative time-to-goal per success
            return float((sum(elements['time_to_goal']) / elements['success_count']))
        elif self._performance_metric == 6:
            # time-to-death per death (ONLY makes sense when pitfalls are enabled)
            return -float(sum(elements['time_to_death']) / elements['death_count'])
        else:
            raise NotImplementedError


if __name__ == '__main__':
    """ Requires pygame
    
    Runs MazeGameSim instance(s) according to user input parameters.
    
    Example Usage:
    python maze.py --tensorboard=True --play-new-levels=True --number-of-iterations=1 --number-of-moves=1000 --area-dimensions 6 6
    or
    python maze.py --tensorboard=True --enable-pitfalls=True --number-of-iterations=100 --number-of-moves=100000 --area-dimensions 6 6
    or
    python maze.py --tensorboard=True --number-of-iterations=100 --number-of-moves=100000 --area-dimensions 10 10 --render=False
    
    """
    import argparse

    parser = argparse.ArgumentParser(prog="RL4Sys Maze Game Simulator",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model-path', type=str, default=None,
                        help='path to pre-existing model to be loaded by agent')
    parser.add_argument('--tensorboard', type=bool, default=True,
                        help='enable tensorboard logging for training observations and insights.\n' +
                             'Make sure to properly configure tensorboard parameters in config.json before running.')
    parser.add_argument('--maze-path', type=str, default=None,
                        help='path to pre-existing maze file to be loaded')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for random number generation in maze')
    parser.add_argument('--score-type', type=int, default=0,
                        help='0. avg action reward per reward, 1. avg action reward per success, 2. avg action reward per death,\n' +
                             '3. avg action reward per collision, 4. avg action reward per failure, 5. Time-to-Goal, 6. Time-to-Death')
    parser.add_argument('--enable-pitfalls', type=bool, default=True,
                        help='enable pitfall generation in maze. NOTE: increases complexity/convergence difficulty')
    parser.add_argument('--play-new-levels', type=bool, default=False,
                        help='Generate and cycle through new mazes after each episode')
    parser.add_argument('--area-dimensions', nargs='+', type=int, default=(6, 6),
                        help='dimensions of the maze area')
    parser.add_argument('--static-area-dimensions', type=bool, default=True,
                        help='Use static area dimensions for maze generation when playing new levels')
    parser.add_argument('--number-of-iterations', type=int, default=100,
                        help='number of iterations to train the agent per level')
    parser.add_argument('--number-of-moves', type=int, default=100000,
                        help='maximum number of moves allowed per iteration')
    parser.add_argument('--start-server', '-s', dest='algorithm', type=str, default='SAC',
                        help='run a local training server, using a specific algorithm, possible options are "DQN" "PPO" "SAC"')
    parser.add_argument('--render', type=bool, default=False,
                        help='render the pygame maze game environment')
    args, extras = parser.parse_known_args()

    # start training server
    app_dir = os.path.dirname(os.path.abspath(__file__))
    if args.algorithm != 'No Server':
        extras.append('--buf_size')
        extras.append(str(MOVE_SEQUENCE_SIZE * 100))
        rl_training_server = TrainingServer(args.algorithm, MAX_SIZE, FEATURES, extras, app_dir, args.tensorboard)
        print('[maze.py] Created Training Server')

    # load model if applicable
    model_arg = torch.load(args.model_path, map_location=torch.device('cpu')) if args.model_path else None

    # load maze if applicable
    loaded_maze = None
    if args.maze_path:
        maze_array = read_maze_from_file(args.maze_path)
        loaded_maze = MazeGenerator(maze=maze_array)

    args.area_dimensions = (args.area_dimensions[0], args.area_dimensions[1])

    # create simulation environment
    maze_game = MazeGameSim(args.seed, model=model_arg, maze=loaded_maze, area_dimensions=args.area_dimensions,
                            static_area_dimensions=args.static_area_dimensions, enable_pitfalls=args.enable_pitfalls,
                            play_new_levels=args.play_new_levels, performance_metric=args.score_type,
                            render_game=args.render)

    # run simulation
    print(f"[maze.py] Running {args.number_of_iterations} iterations for each level...")
    maze_game.run_application(num_iterations=args.number_of_iterations, num_moves=args.number_of_moves,
                              play_new_levels=args.play_new_levels)
