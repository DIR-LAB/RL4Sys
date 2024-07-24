import numpy as np
import random
import time
import os
import sys

import torch
import pygame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from agent import RL4SysAgent
from training_server import TrainingServer

from utils.plot import get_newest_dataset

GAME_ELEMENTS = {'path': 0, 'wall': 1, 'pitfall': 2, 'start': 3, 'goal': 4}


class MazeGenerator:
    def __init__(self, area_dimensions: tuple[int, int], start_position: tuple[int, int],
                 goal_position: tuple[int, int], pitfall_prob: float = 0.1):
        self._area_dimensions = area_dimensions
        self._start = start_position
        self._goal = goal_position
        self._pitfall_prob = pitfall_prob
        self._maze = np.ones(area_dimensions, dtype=int)  # sets maze to all walls
        self._carve_paths()
        self._place_pitfalls()
        self._mark_positions()

    def _carve_paths(self):
        def _is_within_bounds(x, y):
            return 0 <= x < self._area_dimensions[0] and 0 <= y < self._area_dimensions[1]

        def _carve(x, y, width):
            if width == 1:
                if _is_within_bounds(x, y):
                    self._maze[x, y] = GAME_ELEMENTS['path']
            elif width == 2:
                if _is_within_bounds(x, y) and _is_within_bounds(x + 1, y):
                    self._maze[x, y] = GAME_ELEMENTS['path']
                    self._maze[x + 1, y] = GAME_ELEMENTS['path']

        x, y = self._start
        self._maze[x, y] = GAME_ELEMENTS['path']
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while (x, y) != self._goal:
            random.shuffle(directions)
            for dx, dy in directions:
                width = random.choice([1, 2])
                nx, ny = x + dx * width, y + dy * width
                if _is_within_bounds(nx, ny) and self._maze[nx, ny] == GAME_ELEMENTS['wall']:
                    _carve(x, y, width)
                    x, y = nx, ny
                    break

    def _place_pitfalls(self):
        available_area = [(i, j) for i in range(self._area_dimensions[0]) for j in range(self._area_dimensions[1]) if
                          self._maze[i, j] == 0]
        num_pitfalls = int(len(available_area) * self._pitfall_prob)
        pitfalls = random.sample(available_area, num_pitfalls)
        for (i, j) in pitfalls:
            self._maze[i, j] = GAME_ELEMENTS['pitfall']

    def _mark_positions(self):
        self._maze[self._start] = GAME_ELEMENTS['start']
        self._maze[self._goal] = GAME_ELEMENTS['goal']

    def get(self):
        return self._maze, self._area_dimensions, self._start, self._goal, self._pitfall_prob


class AgentProperties:
    def __init__(self, maze: np.ndarray, start_position: tuple[int, int]):
        self.maze = maze
        self.position = start_position

    def move(self, direction: str):
        x, y = self.position
        if direction == 'up':
            self.position = (x - 1, y)
        elif direction == 'down':
            self.position = (x + 1, y)
        elif direction == 'left':
            self.position = (x, y - 1)
        elif direction == 'right':
            self.position = (x, y + 1)
        return self.position

    def calculate_reward(self, next_position: tuple[int, int]):
        x, y = next_position
        if self.maze[x, y] == GAME_ELEMENTS['wall']:
            return -1
        elif self.maze[x, y] == GAME_ELEMENTS['pitfall']:
            return -1
        elif self.maze[x, y] == GAME_ELEMENTS['goal']:
            return 1
        else:
            return 0

    def draw(self, screen):
        x, y = self.position
        color = (0, 128, 128)
        pygame.draw.circle(screen, color, (x * 20, y * 20, 20, 20))


def write_maze_to_log_dir(maze: np.ndarray, log_dir: str):
    with open(f"{log_dir}", 'w') as f:
        f.write(str(maze))


def read_maze_from_file(file_address: str) -> np.ndarray:
    try:
        with open(f"{file_address}", 'r') as f:
            maze = np.array(f.read())
    except FileNotFoundError:
        print(f"[maze.py] Maze file not found at {file_address}")
    return maze


FEATURES = 3
MAX_QUEUE_SIZE = 12


class MazeGameSim:
    def __init__(self, seed, model=None, maze: MazeGenerator = None, cycle_new_mazes: int = 0,
                 performance_metric: int = 0, tensorboard: bool = False):
        self._area_dimensions = (50, 50)
        self._seed = seed
        self._cycle_new_mazes = cycle_new_mazes
        self._performance_metric = performance_metric
        self._max_agent_moves = 1000

        if maze is None:
            self.set_new_maze()
        else:
            self.maze, self._area_dimensions, self._start_position, self._goal_position, self._pitfall_prob = maze.get()

        self._screen = pygame.display.set_mode((self.maze.shape[0] * 20, self.maze.shape[1] * 20))
        pygame.display.set_caption('Maze Game Simulator')
        self.sim_running = False

        self.rlagent = RL4SysAgent(model=model, tensorboard=tensorboard)
        self.agent_properties = AgentProperties(self.maze, self._start_position)

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
                    color = (0, 0, 255)  # Blue
                elif self.maze[x, y] == GAME_ELEMENTS['goal']:
                    color = (0, 255, 0)  # Green
                pygame.draw.rect(self._screen, color, (x * 20, y * 20, 20, 20))

        self.agent_properties.draw(self._screen)
        pygame.display.flip()

    def run_sim(self, num_iterations: int = 1):
        def reset_agent():
            self.agent_properties = AgentProperties(self.maze, self._start_position)

        def check_win(position):
            return position == self._goal_position

        def check_death(position):
            x, y = position
            return self.maze[x, y] == GAME_ELEMENTS['pitfall']

        def check_move_validity(position):
            x, y = position
            return (self.maze[x, y] != GAME_ELEMENTS['wall'] and 0 <= x < self.maze.shape[0]
                    and 0 <= y < self.maze.shape[1])

        # simulator main loop
        clock = pygame.time.Clock()
        reset = False
        moves = 0
        rew = 0
        for iteration in range(num_iterations):
            print(f"[maze.py] Iteration {iteration}")
            rew = self.agent_properties.calculate_reward(self.agent_properties.position)
            if reset:
                reset_agent()
                moves = 0
            while moves < self._max_agent_moves:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                self.draw_sim()
                rew = self.agent_properties.calculate_reward(self.agent_properties.position)
                obs, mask = self.build_observation()
                rl4sys_action = self.rlagent.request_for_action(obs, mask, rew)
                current_position = self.agent_properties.position
                next_position = self.agent_properties.move(rl4sys_action.act[0])
                if next_position is not check_move_validity(next_position):
                    next_position
                reward = self.agent_properties.calculate_reward(next_position)
                reset = check_win(next_position) or check_death(next_position)
                moves += 1
                if reset:
                    break
                clock.tick(30)

    def build_observation(self):
        # make obs
        obs_vector = np.zeros(MAX_QUEUE_SIZE * FEATURES, dtype=float)

        agent_x, agent_y = self.agent_properties.position
        obs_vector[0:2] = [agent_x, agent_y]

        goal_x, goal_y = self._goal_position
        obs_vector[2:4] = [goal_x, goal_y]

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        nearby_pitfall_count = 0
        for i, (dx, dy) in enumerate(directions):
            x, y = agent_x + dx, agent_y + dy
            if 0 <= x < self.maze.shape[0] and 0 <= y < self.maze.shape[1]:
                obs_vector[4 + i] = self.maze[x, y]
                if self.maze[x, y] == GAME_ELEMENTS['pitfall']:
                    nearby_pitfall_count += 1
            else:
                obs_vector[4 + i] = -1

        nearby_pitfall_prob = nearby_pitfall_count / 4
        obs_vector[8] = nearby_pitfall_prob

        distance_to_goal = np.linalg.norm(np.array([agent_x, agent_y]) - np.array([goal_x, goal_y]))
        obs_vector[9] = distance_to_goal

        obs_vector = torch.tensor(obs_vector, dtype=torch.float)
        # make mask
        mask = np.zeros(MAX_QUEUE_SIZE, dtype=float)
        mask = torch.tensor(mask, dtype=torch.float)
        return obs_vector, mask

    def set_new_maze(self):
        self._start_position = (random.randint(0, 49), random.randint(0, 49))
        self._goal_position = (random.randint(0, 49), random.randint(0, 49))
        while self._goal_position == self._start_position:
            self._goal_position = (random.randint(0, 49), random.randint(0, 49))
        self._pitfall_prob = random.uniform(0, 1.0)
        self.maze, _, _, _, _ = MazeGenerator(self._area_dimensions, self._start_position, self._goal_position,
                                              self._pitfall_prob).get()

    # def get_average_performance_score(self):
    #
    # def calculate_performance_score(self) -> float:
    #     """
    #
    #     :return:
    #     """
    #     def _performance_score():
    #         if self._performance_metic == 0:
    #             # cumulative reward of episode
    #
    #         elif self._performance_metric == 1:
    #             # episode length
    #
    #         elif self._performance_metric == 2:
    #             # success rate
    #
    #         elif self._performance_metric == 3:
    #             # time-to-goal
    #
    #         elif self._performance_metric == 4:
    #             # collision rate
    #
    #         else:
    #             raise NotImplementedError
    #
    #     sum = 0
    #     return .01


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog="RL4Sys Maze Game Simulation",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model_path', type=str, default=None,
                        help='path to pre-existing model to be loaded by agent')
    parser.add_argument('--tensorboard', type=bool, default=False,
                        help='enable tensorboard logging for training observations and insights')
    parser.add_argument('--new-maze-cycles', type=int, default=0,
                        help='Generate and cycle through new mazes after each episode')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for random number generation')
    parser.add_argument('--score_type', type=int, default=0,
                        help='score type for performance evaluation')
    parser.add_argument('--number-of-iterations', type=int, default=10,
                        help='number of iterations to train the agent')
    parser.add_argument('--start-server', '-s', dest='algorithm', type=str, default='PPO',
                        help='run a local training server, using a specific algorithm')
    args, extras = parser.parse_known_args()

    if args.algorithm != 'No Server':
        extras.append('--buf_size')
        extras.append(str(100))
        rl_training_server = TrainingServer(args.algorithm, MAX_QUEUE_SIZE, FEATURES, extras)
        print('[maze.py] Created Training Server')

    # load model if applicable
    model_arg = torch.load(args.model_path, map_location=torch.device('cpu')) if args.model_path else None

    # create simulation environment
    maze_game = MazeGameSim(args.seed, model=model_arg, cycle_new_mazes=args.new_maze_cycles,
                            performance_metric=args.score_type, tensorboard=args.tensorboard)
    maze_game.run_sim()

    # run simulation
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'log'))
    iters = args.number_of_iterations
    if args.new_maze_cycles != 0:
        cycles = args.new_maze_cycles
        print(f"[maze.py] Running {iters} iterations for {cycles} new maze cycles")
        for cycle in range(0, cycles):
            write_maze_to_log_dir(maze_game.maze, get_newest_dataset(log_dir, return_file_root=True))
            maze_game.run_sim(iters)
            maze_game.set_new_maze()
    else:
        write_maze_to_log_dir(maze_game.maze, get_newest_dataset(log_dir, return_file_root=True))
        print(f"[maze.py] Running {iters} iterations")
        maze_game.run_sim(iters)
