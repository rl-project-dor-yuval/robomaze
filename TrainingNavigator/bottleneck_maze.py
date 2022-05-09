import numpy as np
import sys
sys.path.append('.')
import MazeEnv.MazeEnv as mz
import cv2
import time
import matplotlib.pyplot as plt
from Utils import get_freespace_map


maze_map = - (cv2.imread("TrainingNavigator/maps/EasyBottleneck.png", cv2.IMREAD_GRAYSCALE) / 255) + 1
maze_map = cv2.rotate(maze_map, cv2.cv2.ROTATE_90_CLOCKWISE)

free_space = get_freespace_map(maze_map, 28)  # was 24 before for bottleneck
free_space_rotated = cv2.rotate(free_space, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite("TrainingNavigator/maps/EasyBottleneck_freespace.png", (-free_space_rotated + 1) * 255)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(-maze_map+1, cmap='gray')
ax1.set_title("Maze Map")
ax2.imshow(-free_space+1, cmap='gray')
ax2.set_title("Free Space For Robot")
plt.show()


env = mz.MazeEnv(maze_size=mz.MazeSize.SQUARE10,
                 maze_map=maze_map,
                 tile_size=0.1,
                 start_loc=(5, 1.4),
                 target_loc=(1.7, 8),
                 show_gui=True)  # missing, timeout, rewards

env.reset()

for i in range(10**4):
    action = [0, 1] * 4
    env.step(action)
    env.set_subgoal_marker((2, 2))
    time.sleep(0.05)
