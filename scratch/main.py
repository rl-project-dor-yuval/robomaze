import os, sys
import torch
sys.path.append('../..')

import MazeEnv.MultiTargetMazeEnv as mz
import time
import numpy as np
from Training.Utils import *
start = time.time()


# create environment :
tile_size = 0.1
maze_size = mz.MazeSize.SQUARE10
map_size = np.dot(maze_size, int(1 / tile_size))
maze_map = make_circular_map(map_size, 3 / tile_size)
# maze_map = np.zeros(map_size)
START_LOC = (5, 5)

targets_loc = np.genfromtxt("Training/TestTargets/test_coords_0_6to3.5.csv", delimiter=',')
print(targets_loc)

maze_env = mz.MultiTargetMazeEnv(maze_size=maze_size,
                             maze_map=maze_map,
                             tile_size=tile_size,
                             start_loc=START_LOC,
                             target_loc_list=targets_loc,
                             timeout_steps=500,
                             show_gui=True,
                             xy_in_obs=False)


# run stepper to test and visualize angles
if __name__ == "__main__":

    model = DDPG.load("./Training/logs/StepperV2CloserTarget0.6_MaxVel0.75_TargetEps3.5/best_model.zip")
    model = model.policy.actor

    for tgt_idx in [6, 7, 8, 9, 10]:

        maze_env.reset(target_index=tgt_idx, create_video=False)

        is_done = False
        obs = maze_env.observation_space.sample()
        while is_done is False:

            action, _ = model.predict(obs)
            obs, reward, is_done, _ = maze_env.step(action)

            if reward != 0:
                print(reward)
            time.sleep(1. / 20)
