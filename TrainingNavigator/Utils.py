import sys

from MazeEnv.EnvAttributes import Rewards

sys.path.append("../")
import numpy as np
import cv2
import MazeEnv.MazeEnv as mz
import matplotlib.pyplot as plt
from TrainingNavigator.StepperAgent import StepperAgent
from TrainingNavigator.NavigatorEnv import NavigatorEnv
from stable_baselines3.common.env_checker import check_env
from gym.wrappers.rescale_action import RescaleAction


def blackwhiteswitch(img_path):
    """
    black and white switch for the image
    """
    return - (cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255) + 1


def tf_top_left_to_bottom_left(vec: np.ndarray, size):
    return np.array([vec[0], size - vec[1] - 1])


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


def get_freespace_map_circular_robot(maze_map, robot_diameter):
    assert robot_diameter % 2 == 0, "robot_diameter must be even"
    robot_radius = robot_diameter // 2
    map_h = maze_map.shape[0]
    map_w = maze_map.shape[1]

    freespace_map = np.copy(maze_map)

    window_center = np.array([robot_radius, robot_radius])
    for i in range(map_h - robot_radius):
        for j in range(map_w - robot_radius):
            window = maze_map[i: i+robot_diameter, j: j+robot_diameter]
            window_filled_pixels = np.argwhere(window > 0)

            if any(np.linalg.norm(window_center - window_filled_pixels, axis=1) < robot_radius):
                # center of that window is not free space
                freespace_map[i + robot_radius, j + robot_radius] = 1

    # # too close to the edges is not free space:
    freespace_map[0:robot_radius, :] = freespace_map[map_h-robot_radius: map_h] = 1
    freespace_map[:, 0:robot_radius] = freespace_map[:, map_w-robot_radius:map_w] = 1

    return freespace_map


def get_vanilla_navigator_env(start_loc=(1., 7.5),
                              target_loc=(9, 3),
                              subgoal_epsilon=0.4,
                              subgoal_max_vel=1,
                              show_gui=True,
                              stepper_path=None):
    """
    create a navigator env with the vanilla maze to solve,
    NOT wrapped with RescaleAction
    :param subgoal_max_vel:
    :param subgoal_epsilon:
    :param start_loc: starting location cords
    :param target_loc: target location cords
    :param show_gui: if true, show the gui
    :param stepper_path: path for pt file (pytorch) of actor stepper model
    """
    assert stepper_path is not None, "stepper_path must be provided"
    map_path = "TrainingNavigator/maps/vanilla_map.png"
    maze_map = - (cv2.imread(map_path, cv2.IMREAD_GRAYSCALE) / 255) + 1
    maze_map = maze_map.T

    raise NotImplementedError("Fix to new workspaces")
    env = mz.MazeEnv(maze_size=mz.MazeSize.SQUARE10,
                     maze_map=maze_map,
                     tile_size=0.05,
                     start_loc=start_loc,
                     target_loc=target_loc,
                     xy_in_obs=True,
                     show_gui=show_gui)  # missing, timeout, rewards

    agent = StepperAgent(agent_path=stepper_path)

    return NavigatorEnv(maze_env=env, stepper_agent=agent, epsilon_to_hit_subgoal=subgoal_epsilon,
                        max_vel_in_subgoal=subgoal_max_vel)


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
        plt.clf()
    else:
        plt.show()


def trajectory_to_transitions(trajectory: np.ndarray, rewards_: Rewards, epsilon_to_hit_subgoal: float,):
    """
    convert a tajectory [(x, y), (x, y), ...] to transitions as lists of:
    observations, actions, rewards, next_states, dones.
    :param trajectory: a trajectory of [(x, y), (x, y), ...]
    :param rewards: reward objects that defines the reward (used for idle and goal reward)
    :return: 5 lists
    """
    trajectory_len = trajectory.shape[0]
    start_loc = trajectory[0]
    goal_loc = trajectory[-1]

    rewards = []
    dones = []
    observations = []
    actions = []
    next_observations = []

    # robot starts heading to the goal:
    prev_heading_at_goal = np.arctan2(goal_loc[1] - start_loc[1], goal_loc[0] - start_loc[0])
    for i in range(trajectory_len - 1):
        curr_loc = trajectory[i]
        next_loc = trajectory[i+1]
        dx, dy = next_loc[0] - curr_loc[0], next_loc[1] - curr_loc[1]
        dist_to_goal = np.linalg.norm(next_loc - goal_loc)

        r_action, theta_action = np.sqrt(dx ** 2 + dy ** 2), np.arctan2(dy, dx)
        heading_action = theta_action
        action = np.array((r_action, theta_action, heading_action))
        obs = np.concatenate((curr_loc, [prev_heading_at_goal], goal_loc))
        next_obs = np.concatenate((next_loc, [heading_action], goal_loc))

        observations.append(obs)
        actions.append(action)
        next_observations.append(next_obs)

        if dist_to_goal < epsilon_to_hit_subgoal or i == trajectory_len - 2:
            # last transition
            rewards.append(rewards_.target_arrival)
            dones.append(True)
            break

        rewards.append(rewards_.idle)
        dones.append(False)

        prev_heading_at_goal = heading_action

    return observations, actions, rewards, next_observations, dones

def make_workspace_list(workspaces):
    """
    Create workspace objects list to pass to MazeEnv given workspaces array of start and goal points.
    Since MazeEnv needs start and goal heading in addition, they will be set for each workspace as
    the angle between the start and goal points for convinence.
    :param workspaces: array of workspaces, each workspace is a tuple of start and goal points
    :return: list of workspaces
    """
    workspaces_list = []
    for workspace in workspaces:
        start_loc = workspace[0]
        goal_loc = workspace[-1]
        heading = np.arctan2(goal_loc[1] - start_loc[1], goal_loc[0] - start_loc[0], )

        ws_tuple = (*start_loc, heading, *goal_loc, heading)
        workspaces_list.append(mz.Workspace.from_array(np.array(ws_tuple)))
    return workspaces_list

