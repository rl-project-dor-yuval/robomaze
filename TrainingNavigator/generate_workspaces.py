import sys
sys.path.append('.')

import cv2
import numpy as np
import ntpath
import seaborn as sns
from pathlib import Path
import argparse
from TrainingNavigator.Utils import get_freespace_map

parser = argparse.ArgumentParser()
parser.add_argument('map', type=str, help='maze map')
parser.add_argument('--ant_size', type=int, default=28,
                    help='Size of ant square for free space map.'
                         ' Ant can be ant_size/2 from wall (in maze pixels)')
parser.add_argument('--num_workspaces', type=int, default=1000,
                    help='number of workspaces to generate')
parser.add_argument('--min_distance', type=int, default=15,
                    help='minimum distance between a target and a goal in maze pixels')
parser.add_argument('--ws_dir_name', type=int,
                    help='directory name for workspace images')
args = parser.parse_args()

# load map, create free space map and save it:
maze_map = - (cv2.imread(args.map, cv2.IMREAD_GRAYSCALE) / 255) + 1
maze_map_rotated = cv2.rotate(maze_map, cv2.cv2.ROTATE_90_CLOCKWISE)

free_space_map = get_freespace_map(maze_map, args.ant_size)  # was 24 before for bottleneck
free_space_rotated = cv2.rotate(free_space_map, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
free_space_map_path = args.map.replace('.png', '_freespace.png')
cv2.imwrite(free_space_map_path, (-free_space_map + 1) * 255)

num_of_workspaces = args.num_workspaces
min_distance = args.min_distance

# free_space_map_path = "TrainingNavigator/maps/EasyBottleneck_freespace.png"
workspaces_dir_name = args.ws_dir_name
if workspaces_dir_name is None:
    workspaces_dir_name = ntpath.basename(args.map).split('.')[0]  # take image name without extension

def generate_start_and_goal(points):
    while True:
        start_idx = np.random.choice(points.shape[0], 1)
        goal_idx = np.random.choice(points.shape[0], 1)
        start = np.squeeze(points[start_idx, :])
        goal = np.squeeze(points[goal_idx, :])
        if np.linalg.norm(start - goal) > min_distance:
            return start, goal


Path(f"TrainingNavigator/workspaces/{workspaces_dir_name}").mkdir(parents=True, exist_ok=True)
free_space_points = np.argwhere(free_space_map == 0)

workspaces = []
# generate hard workspaces:
# for i in range(min_num_hard_workspaces):
#     while True:
#         start, goal = generate_start_and_goal(free_space_points)
#         if (start[1] < 30 and goal[1] > 60) or (start[1] > 60 and goal[1] < 30):
#             # they are on different sides of the map
#             workspaces.append((start, goal))
#             break

for i in range(num_of_workspaces):
    workspaces.append(generate_start_and_goal(free_space_points))

# create plot for each workspace and save
for i, w in enumerate(workspaces):
    w_im = np.copy(-free_space_map + 1)
    cv2.circle(w_im, tuple(reversed(w[0])), 1, 0.2, -1)
    cv2.circle(w_im, tuple(reversed(w[1])), 2, 0.4, -1)

    cv2.imwrite(f"TrainingNavigator/workspaces/{workspaces_dir_name}/{i}.png", w_im * 255)

# create plot of all workspaces
colors = sns.color_palette(None, num_of_workspaces)
color_freespace_map = cv2.imread(free_space_map_path, cv2.IMREAD_COLOR) / 255
color_freespace_map = cv2.rotate(color_freespace_map, cv2.cv2.ROTATE_90_CLOCKWISE)

for i, w in enumerate(workspaces):
    color_freespace_map[tuple(w[0])] = color_freespace_map[tuple(w[1])] = colors[i]
cv2.imwrite(f"TrainingNavigator/workspaces/{workspaces_dir_name}/all.png", color_freespace_map * 255)

# save workspaces to csv:
workspaces = np.array(workspaces)
np.save(f"TrainingNavigator/workspaces/{workspaces_dir_name}", workspaces)
