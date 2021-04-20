import MazeEnv as mz
import time
import os

import pybullet as p


maze = mz.MazeEnv(maze_size=mz.MazeSize.SMALL)
maze.reset()

for i in range(10000):
    maze.step(None)
    time.sleep(1. / 240.)
    # time.sleep(1. / 240.)


maze.reset() # has to be called to save video

# for i in range(1000):
#     maze.step(None)
#     time.sleep(1. / 240.)
