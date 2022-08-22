import os, sys
# import torch
import matplotlib.pyplot as plt

from MazeEnv.EnvAttributes import Workspace, MazeSize
from Training.StepperEnv import StepperEnv

sys.path.append('../..')

import Training.StepperEnv as mz
# import MazeEnv.ObstaclesMultiTargetMazeEnv as omz
import time
import numpy as np
from TrainingNavigator.StepperAgent import StepperAgent
from Training.Utils import *

start = time.time()

# create environment :
tile_size = 0.1
maze_size = MazeSize.SQUARE10
map_size = np.dot(maze_size, int(1 / tile_size))
maze_map = make_circular_map(map_size, 5 / tile_size)
# maze_map = np.zeros(map_size)
START_LOC = (5, 5)

workspaces = np.genfromtxt("Training/workspaces/NoHeading/workspaces_06to3_test.csv", delimiter=',')
workspaces = np.concatenate((np.ones((workspaces.shape[0], 2)) * 5,
                             np.zeros((workspaces.shape[0], 1)),
                             workspaces,
                             np.zeros((workspaces.shape[0], 1))), axis=1)
workspaces = Workspace.list_from_multiple_arrays(workspaces)

# env = omz.ObstaclesMultiTargetMazeEnv(maze_size=maze_size,
#                                            maze_map=maze_map,
#                                            tile_size=tile_size,
#                                            start_loc=START_LOC,
#                                            target_loc_list=targets_loc,
#                                            timeout_steps=500,
#                                            show_gui=True,
#                                            xy_in_obs=False)

env = StepperEnv(maze_size=maze_size,
                 maze_map=maze_map,
                 tile_size=tile_size,
                 workspace_list=workspaces,
                 hit_target_epsilon=0.25,
                 timeout_steps=200,
                 show_gui=True,
                 xy_in_obs=False,
                 sticky_actions=8,
                 noisy_robot_initialization=False,
                 done_on_goal_reached=False,
                 robot_type='Bipedal')


i = 0
# run stepper to test and visualize angles
if __name__ == "__main__":

    agent = StepperAgent("Training/logs/StepperV3HeightReward1908_065355_750775/model_19000000.zip")
    actions = []
    for tgt_idx in range(1):
        obs = env.reset(workspace_index=tgt_idx, create_video=True)
        is_done = False
        # while is_done is False:
        while not is_done:
            # action = np.clip(np.random.randn(8), -1, 1)
            # if i > 200:
            #     action = np.array([0.5, 0.5]*4)
            # action[0] = 1
            # action = env.action_space.sample()
            # if (i//10)%2 == 0:
            #     action = - action
            #
            action = agent.step(obs)
            actions.append(action)
            # action = env.action_space.sample()
            # action = [0.1] * 12
            obs, reward, is_done, _ = env.step(action)
            # print(obs)
            if reward != 0:
                pass
                # print(reward)
            # print("step")

            # print(obs[6:9])
            time.sleep(1. / 500)

    actions = np.array(actions)
    plt.hist(actions)
    plt.show()
