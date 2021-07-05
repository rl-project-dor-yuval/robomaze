import MazeEnv as mz
import time
import os
import time
import numpy as np


def make_circular_map(size, radius):
    center = np.divide(size, 2)
    x, y = np.ogrid[:size[0], :size[1]]
    maze_map = np.where(np.sqrt((x - center[0])**2 + (y-center[1])**2) > radius, 1, 0)

    return maze_map


if __name__ == "__main__":
    start = time.time()

    tile_size = 0.1
    maze_size = mz.MazeSize.SQUARE10
    map_size = np.dot(maze_size, int(1/tile_size))
    maze_map = make_circular_map(map_size, 4./tile_size)
    # maze_map = np.zeros(map_size)

    maze = mz.MazeEnv(maze_size=maze_size,
                      maze_map=maze_map,
                      tile_size=tile_size,
                      start_loc=(5, 3),
                      target_loc=np.divide(maze_size, 2),
                      show_gui=True)

    maze.reset(create_video=False)

    for i in range(10 ** 5):
        # action = maze.action_space.sample()
        action = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        obs, reward, is_done, _ = maze.step(action)
        print(obs)
        time.sleep(1./200.)
        # if reward != 0:
        #     print(reward)

    print(time.time() - start)

    maze.reset()  # has to be called to save video



