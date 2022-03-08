import numpy as np
import MazeEnv.MazeEnv as mz
import cv2
import time


def get_freespace_map(maze_map, robot_cube_size):
    # TODO move this to somewhere later
    assert robot_cube_size % 2 == 0, "robot_cube_size must be even"
    dist = robot_cube_size // 2
    map_h = maze_map.shape[0]
    map_w = maze_map.shape[1]

    freespace_map = np.copy(maze_map)

    for i in range(map_h - robot_cube_size):
        for j in range(map_w - robot_cube_size):
            window = maze_map[i: i+robot_cube_size, j: j+robot_cube_size]
            if np.any(window > 0):
                # center of that window is not free space
                freespace_map[i+dist, j+dist] = 1

    # # too close to the edges is not free space:
    freespace_map[0:dist, :] = freespace_map[map_h-dist: map_h] = 1
    freespace_map[:, 0:dist] = freespace_map[:, map_w-dist:map_w] = 1

    return freespace_map


maze_map = - (cv2.imread("bottleneck.png", cv2.IMREAD_GRAYSCALE) / 255) + 1
maze_map = cv2.rotate(maze_map, cv2.cv2.ROTATE_90_CLOCKWISE)

free_space = get_freespace_map(maze_map, 24)
free_space_rotated = cv2.rotate(free_space, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite("bottleneck_freespace.png", (-free_space_rotated + 1) * 255)

exit()
env = mz.MazeEnv(maze_size=mz.MazeSize.SQUARE10,
                 maze_map=maze_map,
                 tile_size=0.1,
                 start_loc=(5, 1.4),
                 target_loc=(2, 8),
                 show_gui=True)  # missing, timeout, rewards

env.reset()

for i in range(10**4):
    action = [0, 1] * 4
    env.step(action)
    time.sleep(0.05)
