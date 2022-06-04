import os, sys
import torch
import matplotlib.pyplot as plt

sys.path.append('../..')

import MazeEnv.MultiTargetMazeEnv as mz
import MazeEnv.ObstaclesMultiTargetMazeEnv as omz
import time
import numpy as np
from TrainingNavigator.StepperAgent import StepperAgent
from Training.Utils import *

start = time.time()

# create environment :
tile_size = 0.1
maze_size = mz.MazeSize.SQUARE10
map_size = np.dot(maze_size, int(1 / tile_size))
maze_map = make_circular_map(map_size, 5 / tile_size)
# maze_map = np.zeros(map_size)
START_LOC = (5, 5)

targets_loc = np.genfromtxt("Training/workspaces/test_coords_0_6to3.csv", delimiter=',')
print(targets_loc)

# maze_env = omz.ObstaclesMultiTargetMazeEnv(maze_size=maze_size,
#                                            maze_map=maze_map,
#                                            tile_size=tile_size,
#                                            start_loc=START_LOC,
#                                            target_loc_list=targets_loc,
#                                            timeout_steps=500,
#                                            show_gui=True,
#                                            xy_in_obs=False)
maze_env = mtmz.MultiTargetMazeEnv(maze_size=maze_size,
                                   maze_map=maze_map,
                                   tile_size=tile_size,
                                   start_loc=START_LOC,
                                   target_loc_list=targets_loc,
                                   hit_target_epsilon=0.25,
                                   timeout_steps=200,
                                   show_gui=True,
                                   xy_in_obs=False,
                                   sticky_actions=5,
                                   noisy_ant_initialization=True)
i=0
# run stepper to test and visualize angles
if __name__ == "__main__":

    agent = StepperAgent("Training/logs/StepperV2eps025-400steps_same_params/model_6000000.zip")
    for tgt_idx in range(20):
        obs = maze_env.reset(target_index=tgt_idx, create_video=False)
        is_done = False
        actions = []
        # while is_done is False:
        while not is_done:
            action = np.clip(np.random.randn(8), -1, 1)
            # if i > 200:
            #     action = np.array([0.5, 0.5]*4)
            # action[0] = 1
            # action = maze_env.action_space.sample()
            # if (i//10)%2 == 0:
            #     action = - action
            #


            obs, reward, is_done, _ = maze_env.step(action)
            # print(obs)
            if reward != 0:
                print(reward)
            time.sleep(1. / 60)

    actions = np.array(actions)
    plt.hist(actions)
    plt.show()
