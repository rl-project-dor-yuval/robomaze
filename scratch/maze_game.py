import time
from typing import List, Optional

import numpy as np

from log_utils import print_and_log
from toms_maze import generate_random_maze, load_maze_from_data, Maze
from start_goals import StartGoal


class MazeGame:
    def __init__(self, config):
        self.config = config
        self.grid_size = config['maze']['grid_size']
        self.open_area_ratio = config['maze']['open_area_ratio']
        self.contagion_factor = config['maze']['contagion_factor']
        self.noise = config['maze']['noise']

        self.maze = None
        self._current_state = None
        self._goal_state = None
        self._last_step_collision = None
        self._last_step_goal = None

        self._test_time_queries = None

    def get_start_goals(self, number_of_pairs, state_in_initial_distribution=True) -> List[StartGoal]:
        start_time = time.time()
        result = []
        while len(result) < number_of_pairs:
            maze = generate_random_maze(self.grid_size, self.open_area_ratio, self.contagion_factor, self.noise)
            t = self._find_maze_s_g(maze, state_in_initial_distribution)
            result.append(t)
        print_and_log(f'collected {number_of_pairs} queries in {time.time() - start_time}')
        return result

    def _find_maze_s_g(self, maze: Maze, state_in_initial_distribution) -> StartGoal:
        if state_in_initial_distribution:
            start = maze.sample_free_state()
            goal = maze.sample_free_state()
        else:
            start = self._get_random_state()
            goal = self._get_random_state()
        context = maze.get_data()
        observation = maze.create_visual_context(self.config['game']['image_size'])
        return StartGoal(start, goal, context, observation)

    @staticmethod
    def get_state_range():
        return ((-1., -1.), (1., 1.))

    @staticmethod
    def _get_random_state():
        low, high = MazeGame.get_state_range()
        return np.random.uniform(low, high)

    def play_straight_walk(self, context, path, get_collisions=False):
        self.maze = load_maze_from_data(context)
        costs, collisions = zip(*[self._get_segment_cost(path[i], path[i+1]) for i in range(len(path)-1)])
        if get_collisions:
            is_collision = np.any(collisions)
            return sum(costs), is_collision
        else:
            return sum(costs)

    def play_straight_walk_bulk(self, contexts, paths, get_collisions=False):
        return [
            self.play_straight_walk(context, path, get_collisions=get_collisions)
            for context, path in zip(contexts, paths)
        ]

    def _get_segment_cost(self, state1, state2):
        if self.maze.is_collision(state1, state2):
            return self.config['game']['collision_cost'], True
        return np.linalg.norm(np.array(state2) - np.array(state1)), False


    def visualize_values(self, results_dir, level, value_prediction_function):
        import os
        from path_helper import init_dir
        init_dir(results_dir)
        test_queries = self._get_test_time_queries(10, 10)
        for world_id in test_queries:
            context, observation, queries = test_queries[world_id]
            maze = load_maze_from_data(context)
            for image_id, (s, g) in enumerate(queries):
                save_path = os.path.join(results_dir, f'{world_id}_{image_id}.png')
                visualize_values_by_goal(level, g, observation, save_path, value_prediction_function, maze)

    def _get_test_time_queries(self, number_of_worlds, queries_per_world):
        if self._test_time_queries is None:
            results = {}
            for world_id in range(number_of_worlds):
                s, g, context, observation = self.get_start_goals(1, True)[0]
                maze = load_maze_from_data(context)
                results[world_id] = [context, observation, [(s, g)]]
                for image_id in range(1, queries_per_world):
                    t = None
                    while t is None:
                        t = self._find_maze_s_g(maze, True, attempts=100)
                    s, g, context, observation = t
                    results[world_id][-1].append((s, g))
            self._test_time_queries = results
        return self._test_time_queries

    def visualize_end_of_level_trajectories(
            self, results_dir, start_goal_pairs, levels_to_path_ids_to_paths, file_name_with_success=False
    ):
        import os
        from path_helper import init_dir

        # plot the predictions of all the levels:
        init_dir(results_dir)

        for path_id in range(len(start_goal_pairs)):
            paths = [levels_to_path_ids_to_paths[l][path_id] for l in levels_to_path_ids_to_paths]
            file_name = f'{path_id}.png'
            if file_name_with_success:
                assert len(paths) == 1
                current_path, cost, is_success = paths[0]
                file_name_prefix = 'success' if is_success else 'failed'
                file_name = f'{file_name_prefix}_{file_name}'
            save_path = os.path.join(results_dir, file_name)
            paths = [current_path for current_path, cost, is_success in paths]
            maze = load_maze_from_data(start_goal_pairs[path_id][2])
            maze.plot(paths=paths, save_path=save_path)


def visualize_values_by_goal(level, goal, observation, save_path, value_prediction_function, maze: Maze):
    from matplotlib import pyplot as plt
    # grid_res = 21
    # grid_res = 42
    grid_res = 81
    x_ = np.linspace(-1, 1, grid_res)
    y_ = np.linspace(-1, 1, grid_res)
    xs, ys, starts, res = [], [], [], []
    for y in y_:
        for x in x_:
            xs.append(x)
            ys.append(y)
            starts.append(np.array([x, y]))
    goals = [goal] * len(starts)
    observations = [observation] * len(starts)
    res = np.array(value_prediction_function(level, starts, goals, observations))
    res = res.reshape(len(y_), len(x_))
    plt.close('all')
    fig = plt.figure(1, dpi=90)
    ax = fig.add_subplot(111)
    maze.plot_obstacles(ax)
    cf = ax.contourf(x_, y_, res)
    ax.plot(goal[0], goal[1], 'bo')
    s_x, s_y = zip(*starts)
    ax.plot(s_x, s_y, 'r--+', linewidth=0.)
    # for s in starts:
    #     ax.plot(s[0], s[1], 'r--+')
    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1., 1.)
    fig.colorbar(cf, ax=ax)

    plt.savefig(save_path, bbox_inches='tight')
    return fig

