import math
import numpy as np
from TrainingNavigator.NavigatorEnv import NavigatorEnv
from MazeEnv.MazeEnv import MazeEnv
import cv2


# load map and trajectory:
maze_map = - (cv2.imread('maps/bottleneck.png', cv2.IMREAD_GRAYSCALE) / 255) + 1
maze_map = cv2.rotate(maze_map, cv2.ROTATE_90_CLOCKWISE)
traj = np.load('workspaces/botttleneck_trajectories.npz')['0'] / 10

# setup environment:
maze_env = MazeEnv(maze_size=(10, 10), maze_map=maze_map, start_loc=traj[0],
                   target_loc=traj[-1], xy_in_obs=True, show_gui=True)
maze_env.reset()
nav_env = NavigatorEnv(maze_env=maze_env)
nav_env.visualize_mode(True)


obs = nav_env.reset()
# nav_env.maze_env.reset(create_video=args.to_vid, video_path="manualVanilla.avi")

for action in traj:
    dx, dy = obs[0] - action[0], obs[1] - action[1]
    r, theta = math.sqrt(dx ** 2 + dy ** 2), math.atan2(dy, dx)
    action = np.array([r, theta])
    obs, reward, _, _ = nav_env.step(action)
    print("reward:", reward)
