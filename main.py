import MazeEnv as mz
import time
import os
import time
import numpy as np

start = time.time()

maze = mz.MazeEnv(start_loc=(6, 6), show_gui=True)
maze.reset(create_video=False)

t_size = 10000
t = np.linspace(0, t_size-1, t_size)
scale = 0.7

a1 = np.sin(t/10) * scale - 0.1
a2 = np.sin(t/10 + np.pi/4) * scale - 0.1
a3 = np.sin(t/10 + np.pi/2) * scale - 0.1
a4 = np.sin(t/10 + np.pi) * scale - 0.1


for i in range(len(t)):
    # action = maze.action_space.sample()
    #action = [a3[i], a1[i],  a3[i], a1[i], a4[i], a2[i], a4[i], a2[i]]
    # action = np.array(action)
    action = np.zeros(8)
    obs, reward, is_done, _ = maze.step(action)

    # if reward != 0:
    #     print(reward)
    time.sleep(1./240.)


print(time.time() - start)

maze.reset()  # has to be called to save video

# for i in range(10000):
#     action = maze.action_space.sample()
#     _, reward, is_done, _ = maze.step(action)
#     time.sleep(1/250.0)

