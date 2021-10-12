import numpy as np
import glob
import os
import MazeEnv.MultiTargetMazeEnv as mtmz
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env


def clear_files(path: str):
    """
    clean the files from path - accepts file patterns as well
    """
    files = glob.glob(path)
    for f in files:
        os.remove(f)


def make_circular_map(size, radius):
    """
    :param size : size of the map (has to be in class mz.MazeSize)
    :param radius : radius of the maze

    :return : bitmap numpy array of the maze to push to MazeEnv
    """
    center = np.divide(size, 2)
    x, y = np.ogrid[:size[0], :size[1]]
    maze_map = np.where(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) > radius, 1, 0)

    return maze_map


def get_multi_targets_circle_envs(radius, targets, timeout_steps, rewards, monitor_dir):
    # create environment :
    tile_size = 0.1
    maze_size = mtmz.MazeSize.SQUARE10
    map_size = np.dot(maze_size, int(1 / tile_size))
    circle_radius = radius
    maze_map = make_circular_map(map_size, circle_radius / tile_size)

    start_loc = (5, 5)

    maze_env = mtmz.MultiTargetMazeEnv(maze_size=maze_size,
                                       maze_map=maze_map,
                                       tile_size=tile_size,
                                       start_loc=start_loc,
                                       target_loc_list=targets,
                                       timeout_steps=timeout_steps,
                                       show_gui=False,
                                       rewards=rewards)
    maze_env = Monitor(maze_env, filename=monitor_dir)

    check_env(maze_env)

    # create separete evaluation environment:
    eval_maze_env = mtmz.MultiTargetMazeEnv(maze_size=maze_size,
                                            maze_map=maze_map,
                                            tile_size=tile_size,
                                            start_loc=start_loc,
                                            target_loc_list=targets,
                                            timeout_steps=timeout_steps,
                                            show_gui=False,
                                            rewards=rewards)
    return maze_env, eval_maze_env
