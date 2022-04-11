import math
import sys

sys.path.append('.')

import time
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from TrainingNavigator.NavigatorEnv import MultiStartgoalNavigatorEnv
from MazeEnv.MazeEnv import MazeEnv
import cv2
from Utils import plot_trajectory
from DDPGMP import DDPGMP
import torch
from MazeEnv.EnvAttributes import Rewards
import wandb
from TrainingNavigator.NavEvaluation import NavEvalCallback

# --- Parameters
config = {
    "run_name": "TestingEvalCallback",
    "show_gui": False,
    "seed": 42 ** 2,
    "train_steps": 10 ** 5,

    "learning_rate": 1e-5,
    "batch_size": 512,
    "buffer_size": 5 * 10 ** 5,
    "exploration_noise_std": 0.1,
    "epsilon_to_subgoal": 0.8,
    "done_on_collision": True,
    "rewards": Rewards(target_arrival=1, collision=-1, fall=-1, idle=-0.01, ),
    "demonstration_path": 'TrainingNavigator/workspaces/botttleneck_trajectories.npz',
    "demo_on_fail_prob": 0.5,

    "max_stepper_steps": 150,
    "max_navigator_steps": 30,
}
# ---

# setup W&B:
wb_run = wandb.init(project="Robomaze-TrainingNavigator", name=config["run_name"],
                    config=config, )
wandb.tensorboard.patch(root_logdir="TrainingNavigator/logs/tb", pytorch=True)

# Set up the training environment
maze_map = - (cv2.imread('TrainingNavigator/maps/bottleneck.png', cv2.IMREAD_GRAYSCALE) / 255) + 1
start_goal_pairs = np.load('TrainingNavigator/workspaces/bottleneck.npy') / 10

maze_env = MazeEnv(maze_size=(10, 10), maze_map=maze_map, start_loc=start_goal_pairs[0][0],
                   target_loc=start_goal_pairs[0][-1], xy_in_obs=True,
                   show_gui=config["show_gui"], )
nav_env = MultiStartgoalNavigatorEnv(start_goal_pairs=start_goal_pairs,
                                     maze_env=maze_env,
                                     epsilon_to_hit_subgoal=config["epsilon_to_subgoal"],
                                     rewards=config["rewards"],
                                     done_on_collision=config["done_on_collision"],
                                     max_stepper_steps=config["max_stepper_steps"],
                                     max_steps=config["max_navigator_steps"], )
nav_env.visualize_mode(False)
nav_env = Monitor(env=nav_env)

# set up separate evaluation environment:
eval_maze_env = MazeEnv(maze_size=(10, 10), maze_map=maze_map, start_loc=start_goal_pairs[0][0],
                        target_loc=start_goal_pairs[0][-1], xy_in_obs=True, show_gui=False)
eval_nav_env = MultiStartgoalNavigatorEnv(start_goal_pairs=start_goal_pairs,
                                          maze_env=eval_maze_env,
                                          epsilon_to_hit_subgoal=config["epsilon_to_subgoal"],
                                          rewards=config["rewards"],
                                          done_on_collision=config["done_on_collision"],
                                          max_stepper_steps=config["max_stepper_steps"],
                                          max_steps=config["max_navigator_steps"], )
nav_env.visualize_mode(False)

# set up model and run:
exploration_noise = NormalActionNoise(mean=np.array([0] * 2),
                                      sigma=np.array([config["exploration_noise_std"]] * 2))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on:', device)

model = DDPGMP(policy="MlpPolicy",
               env=nav_env,
               buffer_size=config["buffer_size"],
               learning_rate=config["learning_rate"],
               batch_size=config["batch_size"],
               action_noise=exploration_noise,
               device=device,
               train_freq=(4, "episode"),
               verbose=0,
               tensorboard_log="./TrainingNavigator/logs/tb",
               learning_starts=150,
               seed=config["seed"],
               demonstrations_path=config["demonstration_path"],
               demo_on_fail_prob=config["demo_on_fail_prob"], )

# from stable_baselines3 import DDPG
#
# model = DDPG(policy="MlpPolicy",
#              env=nav_env,
#              buffer_size=config["buffer_size"],
#              learning_rate=config["learning_rate"],
#              batch_size=config["batch_size"],
#              action_noise=exploration_noise,
#              device=device,
#              train_freq=(4, "episode"),
#              verbose=0,
#              tensorboard_log="./TrainingNavigator/logs/tb",
#              learning_starts=1000,
#              seed=config["seed"], )

callback = NavEvalCallback(dir="./TrainingNavigator/logs/" + config["run_name"],
                           eval_env=eval_nav_env,
                           wandb_run=wb_run,
                           eval_freq=200, )

model.learn(total_timesteps=config["train_steps"], tb_log_name=config["run_name"], callback=callback)

wb_run.finish()
