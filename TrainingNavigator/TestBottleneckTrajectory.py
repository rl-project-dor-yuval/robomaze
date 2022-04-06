import math
import time

import numpy as np
from TrainingNavigator.NavigatorEnv import NavigatorEnv
from MazeEnv.MazeEnv import MazeEnv
import cv2
from Utils import plot_trajectory


def test_trajectory(traj, maze_map, id, create_video=False):
    if create_video:
        plot_trajectory(traj*10, maze_map,
                        save_loc="TrainingNavigator/test_bottleneck_vids/" + str(id) + "._traj_plot.png")
    # traj = traj / 10  # convert to simulation coordinates

    # setup environment:
    maze_env = MazeEnv(maze_size=(10, 10), maze_map=maze_map, start_loc=traj[0],
                       target_loc=traj[-1], xy_in_obs=True, show_gui=False)
    maze_env.reset()
    nav_env = NavigatorEnv(maze_env=maze_env, done_on_collision=False,)
    nav_env.visualize_mode(False)

    obs = nav_env.reset()
    nav_env.maze_env.reset(create_video=create_video,
                           video_path="TrainingNavigator/test_bottleneck_vids/" + str(id) + ".avi")

    for s in traj:
        dx, dy = s[0] - obs[0], s[1] - obs[1]
        r, theta = math.sqrt(dx ** 2 + dy ** 2), math.atan2(dy, dx)
        r = np.clip(r, nav_env.action_space.low[0], nav_env.action_space.high[0])
        # print('action: r =', r, 'theta =', theta)
        action = np.array([r, theta])
        obs, reward, is_done, info = nav_env.step(action)
        # print("reward:", reward)
        if is_done:
            return info['success']

    # finished running the trajectory, but not done yet
    return False

if __name__ == '__main__':
    maze_map = - (cv2.imread('TrainingNavigator/maps/bottleneck.png', cv2.IMREAD_GRAYSCALE) / 255) + 1
    trajectories = np.load('TrainingNavigator/workspaces/botttleneck_trajectories.npz')

    # check hard workspaces:
    hard_ws_success = 0
    for i in range(30):
        res = test_trajectory(trajectories[str(i)], maze_map, i, create_video=True)
        print (f"trajectory {i} success: {res}")
        hard_ws_success += res

    # check the rest of workspaces :
    normal_ws_success = 0
    for i in range(30, 100):
        res = test_trajectory(trajectories[str(i)], maze_map, i, create_video=True)
        print (f"trajectory {i} success: {res}")
        normal_ws_success += res

    print(f"hard workspaces success: {hard_ws_success}/30")
    print(f"total workspaces success: {hard_ws_success + normal_ws_success}/100")
