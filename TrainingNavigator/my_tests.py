import math
import sys
import torch

sys.path.append('..')
from TrainingNavigator.NavigatorEnv import MultiWorkspaceNavigatorEnv
from MazeEnv.MazeEnv import MazeEnv
import cv2
import numpy as np
import time
from TrainingNavigator.TD3MP import TD3MP
from TrainingNavigator.TestModifyFailureStates import get_scaled_action
from TrainingNavigator.Utils import make_workspace_list


def compute_fake_action(curr_loc, new_loc):
    # works for observations with velocities as well
    dx, dy = new_loc[0] - curr_loc[0], new_loc[1] - curr_loc[1]
    r, theta = math.sqrt(dx ** 2 + dy ** 2), math.atan2(dy, dx)
    rotation = theta

    return np.array([r, theta, rotation])


workspaces = np.load('TrainingNavigator/workspaces/2bedroom/workspaces.npy') / 3
workspaces = make_workspace_list(workspaces)
maze_map = - (cv2.imread('TrainingNavigator/workspaces/2bedroom/2bedroom.png', cv2.IMREAD_GRAYSCALE) / 255) + 1

demos = np.load("TrainingNavigator/workspaces/2bedroom/trajectories_train.npz")

maze_env = MazeEnv(maze_size=(40, 40), maze_map=maze_map, tile_size=1. / 3., show_gui=True, tracking_recorder=True,)

nav_env = MultiWorkspaceNavigatorEnv(workspaces,
                                     maze_env=maze_env,
                                     max_stepper_steps=100,
                                     max_steps=30,
                                     epsilon_to_hit_subgoal=0.25,
                                     done_on_collision=False,
                                     wall_hit_limit=1000,
                                     stepper_radius_range=(0.3, 2),
                                     normalize_observations=False,
                                     stepper_agent='TrainingNavigator/StepperAgents/AntWithHeading.pt',)
nav_env.visualize_mode(True, fps=120)

# nav_agent_path = 'TrainingNavigator/logs/bufferSize33k_demoprob02/saved_model/model_250000.zip'
# agent = TD3MP.load(nav_agent_path, env=nav_env).policy.actor.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in range(1, 20):
    obs = nav_env.reset(i, create_video=False)
    ws_id = nav_env.curr_ws_index
    demo_traj = demos[str(ws_id)]

    done = False
    step = 0
    action = None
    while not done:
        if step <= demo_traj.shape[0]:
            action = compute_fake_action(demo_traj[step], demo_traj[step + 1])
        obs, reward, done, info = nav_env.step(action)
        if reward != 0:
            print(reward)
        step += 1
        if step >= 20:
            break
    # time.sleep(2)
