import os.path
import random
import sys

from matplotlib import pyplot as plt

from TrainingNavigator.generate_workspaces import prepeare_freespace

sys.path.append('.')

import cv2
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse
from TrainingNavigator.Utils import get_freespace_map, get_freespace_map_circular_robot, blackwhiteswitch

parser = argparse.ArgumentParser()
parser.add_argument('map', type=str, help='map path. everything is created in the same dir')
parser.add_argument('--ant_size', type=int, default=6,
                    help='Size of ant square for free space map.'
                         ' Ant can be ant_size/2 from wall (in maze pixels)')
parser.add_argument('--validation_ratio', type=int, default=0.1,
                    help='ratio of validation workspaces from all workspaces')
parser.add_argument('--test_ratio', type=int, default=0.3,
                    help='ratio of test workspaces from all workspaces')

args = parser.parse_args()
workspaces_dir_name = os.path.dirname(args.map)

free_space_map, free_space_map_path = prepeare_freespace(args.map, args.ant_size)

plt.imshow(free_space_map, cmap='gray')
key_points = plt.ginput(n=0, timeout=0)

# reverse X and Y to match old code
key_points = np.array(key_points)[:, [1, 0]]

freespace_map_with_keypoints = cv2.imread(free_space_map_path, cv2.IMREAD_COLOR) / 255
for p in key_points:
    cv2.circle(freespace_map_with_keypoints, (int(p[1]), int(p[0])), 1, (0, 0, 255), -1)
cv2.imwrite(workspaces_dir_name + f'/keypoints.png', freespace_map_with_keypoints * 255)

workspaces = []
for start in key_points:
    for goal in key_points:
        if start[0] != goal[0] and start[1] != goal[1]:
            workspaces.append((start, goal))

random.shuffle(workspaces)
workspaces = np.array(workspaces)

test_last_idx = int(len(workspaces) * args.test_ratio)
validation_last_idx = int(len(workspaces) * (args.validation_ratio + args.test_ratio))

workspaces_test = workspaces[:test_last_idx]
workspaces_validation = workspaces[test_last_idx:validation_last_idx]
workspaces_train = workspaces[validation_last_idx:]

np.save(workspaces_dir_name + f'/workspaces.npy', workspaces_train)
np.save(workspaces_dir_name + f'/validation_workspaces.npy', workspaces_validation)
np.save(workspaces_dir_name + f'/test_workspaces.npy', workspaces_test)

print(workspaces_test.shape, workspaces_validation.shape, workspaces_train.shape)

