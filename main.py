import MazeEnv as mz
import time
import os

maze = mz.MazeEnv(maze_size=mz.MazeSize.SMALL,)
maze.reset(create_video=True)

for i in range(50):
    maze.step(None)
    time.sleep(1. / 240.)


maze.reset()  # has to be called to save video

for i in range(50):
    maze.step(None)
    time.sleep(1. / 240.)

