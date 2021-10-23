import sys

sys.path.append('..')

import cv2
import numpy as np
import MazeEnv.MazeEnv as mz
import time
from MazeEnv.Ant import scale as scale

np_maze_map = - (cv2.imread("vanilla_map.png", cv2.IMREAD_GRAYSCALE) / 255) + 1
m_size = mz.MazeSize.SQUARE10

env = mz.MazeEnv(maze_size=m_size,
                 maze_map=np_maze_map,
                 tile_size=0.05,
                 start_loc=(1, 7.5),
                 target_loc=(9, 3),
                 show_gui=True)  # missing rewards, timeout, xy_in_obs

env.reset()

for i in range(10000):
    env.step([1]*8)
    time.sleep(1/200.)
