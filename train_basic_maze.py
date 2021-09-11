import MazeEnv as mz
import time

import numpy as np
from stable_baselines3 import DDPG


from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
plot_results()
load_results()

def make_circular_map(size, radius):
    center = np.divide(size, 2)
    x, y = np.ogrid[:size[0], :size[1]]
    maze_map = np.where(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) > radius, 1, 0)

    return maze_map

if __name__ == "__main__":
    start = time.time()

    # create environment :
    tile_size = 0.1
    maze_size = mz.MazeSize.SQUARE10
    map_size = np.dot(maze_size, int(1 / tile_size))
    maze_map = make_circular_map(map_size, 3. / tile_size)
    # maze_map = np.zeros(map_size)
    maze_env = mz.MazeEnv(maze_size=maze_size,
                          maze_map=maze_map,
                          tile_size=tile_size,
                          start_loc=(5, 3),
                          target_loc=np.divide(maze_size, 2),
                          timeout_steps=100,
                          show_gui=False)
    maze_env.reset()

    # create model:
    model = DDPG(policy="MlpPolicy",
                 env=maze_env,
                 buffer_size=10**5,  # smaller buffer for small task
                 device='cuda',
                 verbose=1)

    model.learn(total_timesteps=10 ** 4,
                eval_env=maze_env,
                eval_freq=1000,
                n_eval_episodes=20,
                eval_log_path="logs/train_basic_maze"
                )

    print("time", time.time() - start)

    # model.save("models/basic_maze")

    # maze_env.reset()  # has to be called to save video


