import MazeEnv as mz
import time
import os

ss = {"start_loc": (0, 0, 0), "target_loc": (3, 3, 0)}
maze = mz.MazeEnv(start_state=ss)
maze.reset()
s = {mz.MazeSize.SMALL, mz.MazeSize.MEDIUM, mz.MazeSize.LARGE}

for i in range(10000):
    maze.step(None)
    time.sleep(1. / 240.)