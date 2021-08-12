import MazeEnv as mz
import time
import os
import numpy as np
from stable_baselines3 import DDPG


def make_circular_map(size, radius):
    center = np.divide(size, 2)
    x, y = np.ogrid[:size[0], :size[1]]
    maze_map = np.where(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) > radius, 1, 0)

    return maze_map


if __name__ == "__main__":
    start = time.time()

    tile_size = 0.1
    maze_size = mz.MazeSize.SQUARE10
    map_size = np.dot(maze_size, int(1 / tile_size))
    maze_map = make_circular_map(map_size, 3. / tile_size)
    # maze_map = np.zeros(map_size)

    maze_env = mz.MazeEnv(maze_size=maze_size,
                          maze_map=maze_map,
                          tile_size=tile_size,
                          start_loc=(5, 3.4),
                          target_loc=np.divide(maze_size, 2),
                          timeout_steps=500,
                          show_gui=True)
    maze_env.reset()
    model = DDPG(policy="MlpPolicy", env=maze_env, )
    model.learn(total_timesteps=10 ** 5)


    print("time", time.time() - start)

    # maze_env.reset()  # has to be called to save video
