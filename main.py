import os, sys

sys.path.append('..')

import MazeEnv.MazeEnv as mz
import time
import numpy as np

start = time.time()


def make_circular_map(size, radius):
    center = np.divide(size, 2)
    x, y = np.ogrid[:size[0], :size[1]]
    maze_map = np.where(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) > radius, 1, 0)

    return maze_map


# create environment :
tile_size = 0.1
maze_size = mz.MazeSize.SQUARE10
map_size = np.dot(maze_size, int(1 / tile_size))
maze_map = make_circular_map(map_size, 4 / tile_size)
# maze_map = np.zeros(map_size)
TARGET_LOC = (3, 5)
START_LOC = (5, 5)

for _ in range(10):
    maze = mz.MazeEnv(maze_size=maze_size,
                      maze_map=maze_map,
                      tile_size=tile_size,
                      start_loc=START_LOC,
                      target_loc=TARGET_LOC,
                      timeout_steps=500,
                      show_gui=True, )
    maze.reset(create_video=False)
    time.sleep(3)
    for i in range(3 * 10 ** 2):
        action = [1] * 8
        # if i > 2 * 10 ** 2:
        #     action = [1]*8
        obs, reward, is_done, _ = maze.step(action)

        # if reward != 0:
        #     print(reward)
        time.sleep(1. / 240.)

print(time.time() - start)

maze.reset()  # has to be called to save video

# for i in range(10000):
#     action = maze.action_space.sample()
#     _, reward, is_done, _ = maze.step(action)
#     time.sleep(1/250.0)
