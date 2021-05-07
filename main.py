import MazeEnv as mz
import time
import os
import time

maze = mz.MazeEnv(timeout_steps=500)
maze.reset(create_video=False)

for i in range(1000):
    action = maze.action_space.sample()
    _, reward, is_done, _ = maze.step(action)
    print("reward:", reward)
    print("is done", is_done)
    time.sleep(1./500.)

maze.reset()  # has to be called to save video

for i in range(10000):
    action = maze.action_space.sample()
    _, reward, is_done, _ = maze.step(action)
    time.sleep(1/250.0)

