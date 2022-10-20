import math
import sys
import torch

from MazeEnv.EnvAttributes import Rewards

sys.path.append('..')
from TrainingNavigator.NavigatorEnv import MultiWorkspaceNavigatorEnv
from MazeEnv.MazeEnv import MazeEnv
import cv2
import numpy as np
import time
from TrainingNavigator.TD3MP import TD3MP
from TrainingNavigator.TestModifyFailureStates import get_scaled_action
from TrainingNavigator.Utils import make_workspace_list, trajectory_to_transitions_with_heading


def compute_fake_action(curr_loc, new_loc):
    # works for observations with velocities as well
    dx, dy = new_loc[0] - curr_loc[0], new_loc[1] - curr_loc[1]
    r, theta = math.sqrt(dx ** 2 + dy ** 2), math.atan2(dy, dx)
    rotation = theta

    return np.array([r, theta, rotation])


workspaces = np.load('TrainingNavigator/workspaces/S-narrow/workspaces.npy') / 10
workspaces = make_workspace_list(workspaces)
maze_map = - (cv2.imread('TrainingNavigator/workspaces/S-narrow/S-narrow.png', cv2.IMREAD_GRAYSCALE) / 255) + 1

# demos = np.load("TrainingNavigator/workspaces/HugeMazeLight/trajectories_test.npz")

maze_env = MazeEnv(maze_size=(15, 15), maze_map=maze_map, tile_size=1. / 10., show_gui=True, tracking_recorder=True,)

nav_env = MultiWorkspaceNavigatorEnv(workspaces,
                                     maze_env=maze_env,
                                     max_stepper_steps=100,
                                     max_steps=10,
                                     epsilon_to_hit_subgoal=0.25,
                                     done_on_collision=False,
                                     wall_hit_limit=9999,
                                     stepper_radius_range=(0.3, 2.5),
                                     normalize_observations=True,
                                     stepper_agent='TrainingNavigator/StepperAgents/AntWithHeading.pt',)
# nav_env.visualize_mode(False, fps=1000)

# nav_agent_path = 'TrainingNavigator/logs/bufferSize33k_demoprob02/saved_model/model_250000.zip'
# agent = TD3MP.load(nav_agent_path, env=nav_env).policy.actor.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

agent = torch.load('TrainingNavigator/models/S-narrowImitationAgent.pt')


def unnormalize_action(action, action_space):
    # actions are normalized to [-1, 1], get them back to original range
    low, high = action_space.low, action_space.high
    return (action + 1) * (high - low) / 2 + low

dummy_rewards = Rewards()
demos = np.load("TrainingNavigator/workspaces/S-narrow/trajectories_train.npz")

for i in range(1, 20):
    obs = nav_env.reset(i, create_video=False)
    ws_id = nav_env.curr_ws_index
    traj_observations, traj_actions, _, _, _ = \
        trajectory_to_transitions_with_heading(demos[str(i)], dummy_rewards, 0.25)
    # demo_traj = demos[str(ws_id)]

    done = False
    step = 0
    action = None
    while not done:
        action_agent = agent(torch.from_numpy(obs).to(device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
        action = unnormalize_action(action_agent, nav_env.action_space)
        # traj_action = traj_actions[step]
        # traj_action_partial_normalized = traj_action / np.pi
        # expected_obs = traj_observations[step + 1]
        obs, reward, done, info = nav_env.step(action)
        if reward != 0:
            print(reward)
        step += 1
        if step >= 20:
            break
    # time.sleep(2)
