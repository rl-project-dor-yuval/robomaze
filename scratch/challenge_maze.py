import sys

sys.path.append('..')

import cv2
import numpy as np
import MazeEnv.MazeEnv as mz
import time
from MazeEnv.Ant import scale as scale

np_maze_map = - (cv2.imread("data4.png", cv2.IMREAD_GRAYSCALE) / 255) + 1
m_size = mz.MazeSize.SQUARE10

# translate mz.MazeSize
start_image = cv2.imread("data5.png", cv2.IMREAD_GRAYSCALE) / 255
goal_image = cv2.imread("data6.png", cv2.IMREAD_GRAYSCALE) / 255


def get_location_from_npdata(data: np.ndarray, mazeSize: mz.MazeSize):
    """
    get the x,y values given a numpy array that describes a maze map, whereas white is free space and black is the
    desired location
    :param data: numpy array of image that describes the location of a certain point in the maze
    :param mazeSize: the maze size in the simulation
    :return: x, y in mazeSize resolution
    """
    x, y = np.unravel_index(np.argmin(data), data.shape)
    x_mz, y_mz = scale(x, np_maze_map.shape[0], 0, mazeSize[0], 0), scale(y, np_maze_map.shape[1], 0, mazeSize[1], 0)
    return x_mz, y_mz


xs, ys = get_location_from_npdata(start_image, m_size)
xg, yg = get_location_from_npdata(goal_image, m_size)

env = mz.MazeEnv(maze_size=m_size,
                 maze_map=np_maze_map,
                 tile_size=0.05,
                 start_loc=(xs, ys),
                 target_loc=(xg, yg),
                 show_gui=True)  # missing rewards, timeout,

env.reset()

for i in range(10000):
    env.step([1] * 8)
    time.sleep(1 / 200.)
