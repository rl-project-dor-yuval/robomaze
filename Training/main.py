import os, sys

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
maze_map = make_circular_map(map_size, 2.9 / tile_size)
# maze_map = np.zeros(map_size)
START_LOC = (5, 5)

targets_loc = np.genfromtxt("TestTargets/test_coords.csv", delimiter=',')
print(targets_loc)

maze = mz.MultiTargetMazeEnv(maze_size=maze_size,
                             maze_map=maze_map,
                             tile_size=tile_size,
                             start_loc=START_LOC,
                             target_loc_list=targets_loc,
                             timeout_steps=500,
                             action_force=1000,
                             show_gui=True,)

for i in range(1):
    maze.reset(target_index=i, create_video=False)

    for i in range(200):
        action = [-1, 1] * 4
        if i % 50 > 25:
            action = [1, -1]*4

        # action = np.random.random(8)*2 - 1
        obs, reward, is_done, _ = maze.step(action)

        # if reward != 0:
        #     print(reward)
        time.sleep(1. / 100)

print(time.time() - start)

maze.reset()  # has to be called to save video

# for i in range(10000):
#     action = maze.action_space.sample()
#     _, reward, is_done, _ = maze.step(action)
#     time.sleep(1/250.0)
