import MazeEnv.MazeEnv as mz
import time
import os
import time
import numpy as np

start = time.time()

maze = mz.MazeEnv(start_loc=(1, 1), target_loc=(9, 5), show_gui=True, maze_size=mz.MazeSize.SQUARE10)

for _ in range(4):
    maze.reset(create_video=False)
    time.sleep(3)
    for i in range(3 * 10 ** 2):
        # action = maze.action_space.sample()
        # action = [a3[i], a1[i],  a3[i], a1[i], a4[i], a2[i], a4[i], a2[i]]
        # action = np.array(action)
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
