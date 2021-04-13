import MazeEnv as mz
import time
import os

maze = mz.MazeEnv()
maze.reset()

for i in range(10000):
    maze.step(None)
    time.sleep(1. / 240.)