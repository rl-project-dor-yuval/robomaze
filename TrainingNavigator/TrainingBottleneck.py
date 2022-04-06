import math
import sys

sys.path.append('.')

import time
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from TrainingNavigator.NavigatorEnv import MultiStartgoalNavigatorEnv
from MazeEnv.MazeEnv import MazeEnv
import cv2
from Utils import plot_trajectory
from DDPGMP import DDPGMP
import torch
from MazeEnv.EnvAttributes import Rewards

# --- Parameters
RUN_NAME = "DDPGMP_SomeImprovements"
SHOW_GUI = True
SEED = 42 ** 2
TRAIN_STEPS = 10**7

LEARNING_RATE = 1e-5
BUFFER_SIZE = 5 * 10 ** 5
EXPLORATION_NOISE_STD = 0.1
EPSILON_TO_SUBGOAL = 0.8
REWARDS = Rewards(target_arrival=1, collision=-1, fall=-1, idle=-0.01,)
DEMONSTRATION_PATH = 'TrainingNavigator/workspaces/botttleneck_trajectories.npz'
DEMO_ON_FAIL_PROB = 0.5
# ---

maze_map = - (cv2.imread('TrainingNavigator/maps/bottleneck.png', cv2.IMREAD_GRAYSCALE) / 255) + 1
start_goal_pairs = np.load('TrainingNavigator/workspaces/bottleneck.npy') / 10

maze_env = MazeEnv(maze_size=(10, 10), maze_map=maze_map, start_loc=start_goal_pairs[0][0],
                   target_loc=start_goal_pairs[0][-1], xy_in_obs=True, show_gui=SHOW_GUI,)
nav_env = MultiStartgoalNavigatorEnv(start_goal_pairs=start_goal_pairs,
                                     maze_env=maze_env,
                                     epsilon_to_hit_subgoal=EPSILON_TO_SUBGOAL,
                                     rewards=REWARDS,)
nav_env.visualize_mode(False)

exploration_noise = NormalActionNoise(mean=np.array([0] * 2), sigma=np.array([EXPLORATION_NOISE_STD] * 2))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on:', device)

model = DDPGMP(policy="MlpPolicy",
               env=nav_env,
               buffer_size=BUFFER_SIZE,
               learning_rate=LEARNING_RATE,
               action_noise=exploration_noise,
               device=device,
               train_freq=(8, "episode"),
               verbose=0,
               tensorboard_log="./TrainingNavigator/logs/tb",
               learning_starts=16,
               seed=SEED,
               demonstrations_path=DEMONSTRATION_PATH,
               demo_on_fail_prob=DEMO_ON_FAIL_PROB, )

# from stable_baselines3 import DDPG
# model = DDPG(policy="MlpPolicy",
#              env=nav_env,
#              buffer_size=BUFFER_SIZE,
#              learning_rate=LEARNING_RATE,
#              action_noise=exploration_noise,
#              device=device,
#              train_freq=(8, "episode"),
#              verbose=0,
#              tensorboard_log="./TrainingNavigator/logs/tb",
#              learning_starts=16,
#              seed=SEED, )

model.learn(total_timesteps=TRAIN_STEPS, tb_log_name=RUN_NAME)
