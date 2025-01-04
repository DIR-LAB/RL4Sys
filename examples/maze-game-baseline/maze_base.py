import argparse
import gym
from gym import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pygame
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Initialize Pygame
pygame.init()

# Define the game elements
GAME_ELEMENTS = {'path': 0, 'wall': 1, 'pitfall': 2, 'start': 3, 'goal': 4}

class MazeGenerator:
    def __init__(self, area_dimensions=(10, 10), start_position=None,
                 goal_position=None, pitfall_prob=0.1, maze=None):
        if maze is not None:
            self._maze, self._area_dimensions, self._start, self._goal, self._pitfall_prob = self._maze_element_search(maze)
            assert self._area_dimensions[0] > 0 and self._area_dimensions[1] > 0
            assert self._start is not None
            assert self._goal is not None
        else:
            self._maze = np.ones(area_dimensions, dtype=int)  # sets maze to all walls
            self._area_dimensions = area_dimensions

            # Set default start and goal positions if not provided
            if start_position is None:
                self._start = (0, 0)
            else:
                self._start = start_position

            if goal_position is None:
                self._goal = (area_dimensions[0] - 1, area_dimensions[1] - 1)
            else:
                self._goal = goal_position

            # Ensure start and goal positions are within bounds
            max_x, max_y = area_dimensions[0] - 1, area_dimensions[1] - 1
            if not (0 <= self._start[0] <= max_x and 0 <= self._start[1] <= max_y):
                raise ValueError(f"Start position {self._start} is out of maze bounds.")
            if not (0 <= self._goal[0] <= max_x and 0 <= self._goal[1] <= max_y):
                raise ValueError(f"Goal position {self._goal} is out of maze bounds.")

            self._pitfall_prob = pitfall_prob
            self._carve_paths()
            self._place_pitfalls()
            self._mark_positions()

    def _maze_element_search(self, maze):
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
        if num_pitfalls > 0:
            pitfalls = random.sample(available_area, num_pitfalls)
            for (i, j) in pitfalls:
                self._maze[i, j] = GAME_ELEMENTS['pitfall']

    def _mark_positions(self):
        self._maze[self._start] = GAME_ELEMENTS['start']
        self._maze[self._goal] = GAME_ELEMENTS['goal']

        # Ensure the goal position has at least one adjacent path
        goal_x, goal_y = self._goal
        adjacent_positions = [(goal_x - 1, goal_y), (goal_x + 1, goal_y),
                              (goal_x, goal_y - 1), (goal_x, goal_y + 1)]
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
    def __init__(self, maze, start_position, goal_position, play_new_levels=False):
        self.maze = maze
        self.start = start_position
        self.position = start_position
        self.goal = goal_position
        self.play_new_levels = play_new_levels

        self.GOAL_REWARD = 1000 if not self.play_new_levels else 500

    def move(self, direction):
        x, y = self.position
        if direction == 0:  # up
            new_position = (x - 1, y)
        elif direction == 1:  # down
            new_position = (x + 1, y)
        elif direction == 2:  # left
            new_position = (x, y - 1)
        elif direction == 3:  # right
            new_position = (x, y + 1)
        else:
            raise ValueError("Invalid action")
        self.position = new_position
        return new_position

    def calculate_reward(self, previous_position, next_position):
        px, py = previous_position
        nx, ny = next_position

        def granular_move_reward():
            def towards_goal():
                if previous_position == next_position:
                    return None
                goal_x, goal_y = self.goal
                start_x, start_y = self.start
                return (abs(nx - goal_x) + abs(ny - goal_y) < abs(px - goal_x) + abs(py - goal_y)) and \
                    (abs(nx - start_x) + abs(ny - start_y) > abs(px - start_x) + abs(py - start_y))

            if towards_goal():
                return 0.3
            else:
                return -0.1

        def element_reward():
            if not (0 <= nx < self.maze.shape[0] and 0 <= ny < self.maze.shape[1]):
                return -1.0
            elif self.maze[nx, ny] == GAME_ELEMENTS['wall']:
                return -0.3
            elif self.maze[nx, ny] == GAME_ELEMENTS['pitfall']:
                return -2.0
            elif self.maze[nx, ny] == GAME_ELEMENTS['goal']:
                return self.GOAL_REWARD
            else:
                return 0.1

        return granular_move_reward() + element_reward()

    def reset(self):
        self.position = self.start

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, area_dimensions=(10, 10), enable_pitfalls=False, render_mode=False):
        super(MazeEnv, self).__init__()
        self.area_dimensions = area_dimensions
        self.enable_pitfalls = enable_pitfalls
        self.pitfall_prob = 0.1 if enable_pitfalls else 0

        self.render_mode = render_mode

        self._max_episode_steps = 1000  # optional

        # Generate the maze
        self._generate_maze()

        # Define action_space and observation_space
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right

        # Observation vector length is 20 (as per the original code)
        obs_length = 20
        # Since observation can include negative values (e.g., walls are -1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_length,), dtype=np.float32)

        # Initialize other variables
        self.agent_properties = AgentProperties(self.maze, self.start_position, self.goal_position)
        self.steps_beyond_done = None
        self.current_step = 0

        # Pygame initialization for rendering
        if self.render_mode:
            self._init_render()

    def _generate_maze(self):
        self.maze_generator = MazeGenerator(area_dimensions=self.area_dimensions, pitfall_prob=self.pitfall_prob)
        self.maze, self.area_dimensions, self.start_position, self.goal_position, self.pitfall_prob = self.maze_generator.get()
        self.agent_properties = AgentProperties(self.maze, self.start_position, self.goal_position)

    def reset(self):
        # Reset the environment to an initial state and returns an initial observation
        self._generate_maze()
        self.agent_properties.reset()
        self.current_step = 0
        self.steps_beyond_done = None

        if self.render_mode:
            self._init_render()

        # Return initial observation
        observation = self._build_observation(self.agent_properties.position)
        return observation

    def step(self, action):
        # Apply the action, update the environment's state
        self.current_step += 1
        previous_position = self.agent_properties.position
        new_position = self.agent_properties.move(action)

        # Calculate reward
        reward = self.agent_properties.calculate_reward(previous_position, new_position)

        # Check if the episode is done
        done = False
        if new_position == self.goal_position:
            done = True
        elif not self._is_valid_position(new_position):
            # Invalid move, reset to previous position
            self.agent_properties.position = previous_position
            reward += -1.0  # Penalty for invalid move
        elif self.maze[new_position] == GAME_ELEMENTS['pitfall']:
            # Agent fell into a pitfall
            done = True
            reward += -10.0  # Additional penalty for falling into pitfall
        else:
            # Valid move
            pass

        # If max steps exceeded, end the episode
        if self.current_step >= self._max_episode_steps:
            done = True

        # Return observation, reward, done, info
        observation = self._build_observation(self.agent_properties.position)
        info = {}

        if self.render_mode:
            self.render()

        return observation, reward, done, info

    def render(self, mode='human'):
        if not self.render_mode:
            return

        # Clear the screen
        self.screen.fill((0, 0, 0))

        # Draw the maze
        for x in range(self.maze.shape[0]):
            for y in range(self.maze.shape[1]):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                if self.maze[x, y] == GAME_ELEMENTS['path']:
                    color = (255, 255, 255)  # White
                elif self.maze[x, y] == GAME_ELEMENTS['wall']:
                    color = (50, 50, 50)  # Dark Gray
                elif self.maze[x, y] == GAME_ELEMENTS['pitfall']:
                    color = (255, 0, 0)    # Red
                elif self.maze[x, y] == GAME_ELEMENTS['start']:
                    color = (0, 255, 0)    # Green
                elif self.maze[x, y] == GAME_ELEMENTS['goal']:
                    color = (0, 0, 255)    # Blue
                pygame.draw.rect(self.screen, color, rect)

        # Draw the agent
        agent_x, agent_y = self.agent_properties.position
        agent_rect = pygame.Rect(agent_y * self.cell_size, agent_x * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 255, 0), agent_rect)  # Yellow

        # Update the display
        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def _init_render(self):
        self.cell_size = 30  # Size of each cell in pixels
        maze_height = self.maze.shape[0] * self.cell_size
        maze_width = self.maze.shape[1] * self.cell_size
        self.screen = pygame.display.set_mode((maze_width, maze_height))
        pygame.display.set_caption('Maze Environment')

    def close(self):
        if self.render_mode:
            pygame.quit()

    def _is_valid_position(self, position):
        x, y = position
        if not (0 <= x < self.maze.shape[0] and 0 <= y < self.maze.shape[1]):
            return False
        if self.maze[x, y] == GAME_ELEMENTS['wall']:
            return False
        return True

    def _build_observation(self, agent_position):
        # Local observation construction
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

        # Initialize local observation with out-of-bound values (-1)
        local_obs = -np.ones((local_view_size, local_view_size))

        # Fill local observation with maze elements within boundaries
        local_obs[local_obs_x_start:local_obs_x_end, local_obs_y_start:local_obs_y_end] \
            = self.maze[min_x:max_x, min_y:max_y]

        # Agent's local position within local grid
        agent_view_x = half_view_size
        agent_view_y = half_view_size
        local_obs[agent_view_x, agent_view_y] = 9

        # Flatten for observation vector
        flat_local_obs = local_obs.flatten()

        # Global observation construction
        global_obs = np.zeros(11, dtype=float)

        # Agent's position in context to whole maze environment
        global_obs[0:2] = [agent_x, agent_y]

        # Goal position in context to whole maze environment
        goal_x, goal_y = self.goal_position
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
        obs = combined_observation.astype(np.float32)
        return obs

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        q_values = self.fc3(x)
        return q_values

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size=100000, batch_size=64):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        return (
            torch.tensor(np.array(state_batch), dtype=torch.float32),
            torch.tensor(action_batch, dtype=torch.int64),
            torch.tensor(reward_batch, dtype=torch.float32),
            torch.tensor(np.array(next_state_batch), dtype=torch.float32),
            torch.tensor(done_batch, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, render=False):
        self.state_size = state_size
        self.action_size = action_size
        self.render = render

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.target_update_freq = 10  # Update target network every N episodes

        # Networks
        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Replay Buffer
        self.memory = ReplayBuffer(buffer_size=100000, batch_size=self.batch_size)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        states = states.to(self.device)
        actions = actions.unsqueeze(1).to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(states).gather(1, actions)

        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # Compute the expected Q values
            expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    parser = argparse.ArgumentParser(description="Train RL agent on Maze Environment")
    parser.add_argument('--algorithm', type=str, default='SAC', choices=['DQN', 'PPO', 'SAC'],
                        help='Select the algorithm to use: DQN, PPO, SAC')
    parser.add_argument('--render', action='store_false', help='Render the environment using GUI')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    args = parser.parse_args()

    # Initialize the environment
    env = MazeEnv(area_dimensions=(6, 6), enable_pitfalls=True, render_mode=args.render)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Select the agent based on the algorithm argument
    if args.algorithm == 'DQN':
        from algorithms_baseline.DQN_baseline import DQNAgent
        agent = DQNAgent(state_size, action_size)
    elif args.algorithm == 'PPO':
        from algorithms_baseline.PPO_baseline import PPOAgent
        agent = PPOAgent(state_size, action_size)
    elif args.algorithm == 'SAC':
        from algorithms_baseline.SAC_baseline import SACAgent
        agent = SACAgent(state_size, action_size)
    else:
        print(f"Unsupported algorithm: {args.algorithm}")
        sys.exit(1)

    num_episodes = args.episodes
    max_steps = env._max_episode_steps

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            if args.render:
                env.render()

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            agent.store_transition(state, action, reward, next_state, done)
            agent.optimize_model()

            state = next_state

            if done:
                break

        agent.update()

        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}")

    env.close()

if __name__ == '__main__':
    main()
