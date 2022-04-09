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
from DDPGMP import DDPGMP, CustomTD3Policy
import torch
from MazeEnv.EnvAttributes import Rewards
import wandb
from wandb.integration.sb3 import WandbCallback


# --- Parameters
config = {
    "run_name": "TestingW&BFromRemote",
    "show_gui": True,
    "seed": 42 ** 2,
    "train_steps": 10 ** 2,

    "learning_rate": 1e-5,
    "batch_size": 1024,
    "buffer_size": 5 * 10 ** 5,
    "actor_arch": [64, 64],
    "critic_arch": [64, 64],
    "exploration_noise_std": 0.1,
    "epsilon_to_subgoal": 0.8,
    "done_on_collision": True,
    "rewards": Rewards(target_arrival=1, collision=-0.01, fall=-0.01, idle=-0.01, ),
    "demonstration_path": 'TrainingNavigator/workspaces/botttleneck_trajectories.npz',
    "demo_on_fail_prob": 0.5,

    "max_stepper_steps": 150,
    "max_navigator_steps": 30,
}

# wb_run = wandb.init(project="Robomaze-TrainingNavigator", name=config["run_name"],
#                     config=config, sync_tensorboard=True)
# wb_callback = WandbCallback(model_save_path=f"TrainingNavigator/checkpoints/{config['run_name']}",
#                             model_save_freq=config["train_steps"] // 10,
#                             gradient_save_freq=config["train_steps"] // 10,)

MAZE_LENGTH = 10
maze_map = - (cv2.imread('TrainingNavigator/maps/bottleneck.png', cv2.IMREAD_GRAYSCALE) / 255) + 1
start_goal_pairs = np.load('TrainingNavigator/workspaces/bottleneck.npy') / 10

maze_env = MazeEnv(maze_size=(MAZE_LENGTH, MAZE_LENGTH), maze_map=maze_map, start_loc=start_goal_pairs[0][0],
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

exploration_noise = NormalActionNoise(mean=np.array([0] * 2),
                                      sigma=np.array([config["exploration_noise_std"]] * 2))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on:', device)

policy_kwargs = dict(net_arch=dict(pi=config["actor_arch"], qf=config["critic_arch"]))
model = DDPGMP(policy=CustomTD3Policy,
               env=nav_env,
               buffer_size=config["buffer_size"],
               learning_rate=config["learning_rate"],
               batch_size=config["batch_size"],
               action_noise=exploration_noise,
               device=device,
               train_freq=(4, "episode"),
               verbose=0,
               tensorboard_log="./TrainingNavigator/logs/tb",
               learning_starts=10,
               seed=config["seed"],
               demonstrations_path=config["demonstration_path"],
               demo_on_fail_prob=config["demo_on_fail_prob"],
               policy_kwargs=policy_kwargs)

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

model.learn(total_timesteps=config["train_steps"])  # , callback=wb_callback)

# wb_run.finish()
