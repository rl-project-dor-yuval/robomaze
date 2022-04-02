import sys
sys.path.append("../")
import numpy as np
import cv2
import MazeEnv.MazeEnv as mz
import matplotlib.pyplot as plt
from TrainingNavigator.StepperAgent import StepperAgent
from TrainingNavigator.NavigatorEnv import NavigatorEnv
from stable_baselines3.common.env_checker import check_env
from gym.wrappers.rescale_action import RescaleAction


def get_freespace_map(maze_map, robot_cube_size):
    # TODO move this to somewhere later
    assert robot_cube_size % 2 == 0, "robot_cube_size must be even"
    dist = robot_cube_size // 2
    map_h = maze_map.shape[0]
    map_w = maze_map.shape[1]

    freespace_map = np.copy(maze_map)

    for i in range(map_h - robot_cube_size):
        for j in range(map_w - robot_cube_size):
            window = maze_map[i: i+robot_cube_size, j: j+robot_cube_size]
            if np.any(window > 0):
                # center of that window is not free space
                freespace_map[i+dist, j+dist] = 1

    # # too close to the edges is not free space:
    freespace_map[0:dist, :] = freespace_map[map_h-dist: map_h] = 1
    freespace_map[:, 0:dist] = freespace_map[:, map_w-dist:map_w] = 1

    return freespace_map


def get_vanilla_navigator_env(start_loc=(1., 7.5), target_loc=(9, 3), show_gui=True):
    """
    create a navigator env with the vanilla maze to solve,
    NOT wrapped with RescaleAction
    """
    map_path = "maps/vanilla_map.png"
    maze_map = - (cv2.imread(map_path, cv2.IMREAD_GRAYSCALE) / 255) + 1
    maze_map = maze_map.T

    env = mz.MazeEnv(maze_size=mz.MazeSize.SQUARE10,
                     maze_map=maze_map,
                     tile_size=0.05,
                     start_loc=start_loc,
                     target_loc=target_loc,
                     xy_in_obs=True,
                     show_gui=show_gui)  # missing, timeout, rewards

    agent_path = "StepperAgent.pt"
    agent = StepperAgent(agent_path)

    return NavigatorEnv(maze_env=env, stepper_agent=agent, )


def get_vanilla_navigator_env_scaled(start_loc=(1., 7.5), target_loc=(9, 3), show_gui=True):
    """
    create a navigator env with the vanilla maze to solve,
    action space is scaled to [-1, 1]
    """
    env = get_vanilla_navigator_env(start_loc, target_loc, show_gui)
    return RescaleAction(env, -1, 1)


def test_navigator_envrionment():
    """
    check if the implementation of NavigatorEnv is valid using sb3 check_env()
    """

    check_env(get_vanilla_navigator_env_scaled())


def plot_trajectory(trajectory, map, save_loc=None):
    """
    plot the trajectory in the map
    """
    plt.imshow(map, cmap='gray')
    plt.scatter( trajectory[:, 1], trajectory[:, 0], c='b')
    if save_loc is not None:
        plt.savefig(save_loc)
    else:
        plt.show()

