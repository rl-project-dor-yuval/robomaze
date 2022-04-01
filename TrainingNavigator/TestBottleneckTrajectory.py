import math
import time

import numpy as np
from TrainingNavigator.NavigatorEnv import NavigatorEnv
from MazeEnv.MazeEnv import MazeEnv
import cv2
from Utils import plot_trajectory


# load map and trajectory:
maze_map = - (cv2.imread('maps/bottleneck.png', cv2.IMREAD_GRAYSCALE) / 255) + 1
traj = np.load('workspaces/botttleneck_trajectories.npz')['0']
plot_trajectory(traj, maze_map)
traj = traj / 10  # convert to simulation coordinates

# setup environment:
maze_env = MazeEnv(maze_size=(10, 10), maze_map=maze_map, start_loc=traj[0],
                   target_loc=traj[-1], xy_in_obs=True, show_gui=True)
maze_env.reset()
nav_env = NavigatorEnv(maze_env=maze_env, epsilon_to_hit_subgoal=0.7)
nav_env.visualize_mode(True)


obs = nav_env.reset()
# nav_env.maze_env.reset(create_video=args.to_vid, video_path="manualVanilla.avi")

for s in traj:
    dx, dy = s[0] - obs[0], s[1] - obs[1]
    r, theta = math.sqrt(dx ** 2 + dy ** 2), math.atan2(dy, dx)
    print('action: r =', r, 'theta =', theta)
    action = np.array([r, theta])
    obs, reward, is_done, _ = nav_env.step(action)
    print("reward:", reward)
    if is_done:
        time.sleep(5)
        break
