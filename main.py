import MazeEnv as mz
import time
import os

import pybullet as p


maze = mz.MazeEnv(maze_size=mz.MazeSize.LARGE)
maze.reset()

for i in range(800):
    maze.step(None)
    time.sleep(1. / 240.)

maze.reset()

for i in range(1000):
    maze.step(None)
    time.sleep(1. / 240.)
