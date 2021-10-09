import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DDPG
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.env_checker import check_env

import matplotlib.pyplot as plt

sys.path.append('..')

import MazeEnv.MazeEnv as mz
from MazeEnv.MazeEnv import Rewards
from Training.Utils import *
from Training.Evaluation import EvalAndSaveCallback, clear_files
import Training.Evaluation

import torch

# Parameters definition
TIMEOUT_STEPS = 300
BUFFER_SIZE = 1000  # smaller buffer for small task
TOTAL_TIME_STEPS = 1000
LEARNING_RATE = 0.001

EVAL_EPISODES = 1
EVAL_FREQ = 200
VIDEO_FREQ = 4

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    REWARDS = Rewards(target_arrival=1, collision=-1, timeout=-0.5, idle=-0.01)

    start = time.time()

    # create environment :
    tile_size = 0.1
    maze_size = mz.MazeSize.SQUARE10
    map_size = np.dot(maze_size, int(1 / tile_size))
    maze_map = make_circular_map(map_size, 4 / tile_size)
    # maze_map = np.zeros(map_size)
    TARGET_LOC = (5, 3)
    START_LOC = np.divide(maze_size, 2)

    maze_env = Monitor(mz.MazeEnv(maze_size=maze_size,
                                  maze_map=maze_map,
                                  tile_size=tile_size,
                                  start_loc=START_LOC,
                                  target_loc=TARGET_LOC,
                                  timeout_steps=TIMEOUT_STEPS,
                                  show_gui=False,
                                  rewards=REWARDS),
                       filename="logs/MultiTargets/results")

    _ = maze_env.reset()
    check_env(maze_env)
    # create separete evaluation environment:
    eval_maze_env = Monitor(mz.MazeEnv(maze_size=maze_size,
                                       maze_map=maze_map,
                                       tile_size=tile_size,
                                       start_loc=START_LOC,
                                       target_loc=np.divide(maze_size, 2),
                                       timeout_steps=TIMEOUT_STEPS,
                                       show_gui=False,
                                       rewards=REWARDS)
                            )
    _ = eval_maze_env.reset()

    # create model:
    model = DDPG(policy="MlpPolicy",
                 env=maze_env,
                 buffer_size=BUFFER_SIZE,
                 learning_rate=LEARNING_RATE,
                 device=device,
                 train_freq=(1, "episode"),
                 verbose=1)

    # create callback for evaluation
    callback = EvalAndSaveCallback(log_dir="logs/MultiTargets",
                                   eval_env=eval_maze_env,
                                   eval_freq=EVAL_FREQ,
                                   eval_episodes=EVAL_EPISODES,
                                   eval_video_freq=VIDEO_FREQ,
                                   verbose=2)

    # clean all movies from the previous run
    clear_files('logs/MultiTargets/*_steps.avi')

    torch.manual_seed(3295)
    model.learn(total_timesteps=TOTAL_TIME_STEPS,
                callback=callback)

    print("time", time.time() - start)
