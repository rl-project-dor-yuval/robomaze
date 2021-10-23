import sys
sys.path.append('..')

import cv2
import numpy as np
import MazeEnv.MazeEnv as mz
import time


maze_map = - (cv2.imread("vanilla_map.png", cv2.IMREAD_GRAYSCALE)/255) + 1
maze_map = maze_map.T

env = mz.MazeEnv(maze_size=mz.MazeSize.SQUARE10,
                 maze_map=maze_map,
                 tile_size=0.05,
                 start_loc=(1, 7.5),
                 target_loc=(9, 3),
                 show_gui=True)  # missing rewards, timeout, xy_in_obs

env.reset()
for i in range(10000):
    env.step([1]*8)
    time.sleep(1/200.)
