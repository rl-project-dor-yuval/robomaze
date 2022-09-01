import os.path
import sys
sys.path.append('.')

import cv2
import numpy as np
import seaborn as sns
from pathlib import Path
import argparse
from TrainingNavigator.Utils import get_freespace_map, get_freespace_map_circular_robot, blackwhiteswitch


def prepeare_freespace(map_path, ant_size):
    # load map, create free space map and save it
    maze_map = blackwhiteswitch(map_path)
    maze_map_rotated = cv2.rotate(maze_map, cv2.cv2.ROTATE_90_CLOCKWISE)

    free_space_map = get_freespace_map_circular_robot(maze_map, ant_size)  # was 24 before for bottleneck
    free_space_rotated = cv2.rotate(free_space_map, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    free_space_map_path = map_path.replace('.png', '_freespace.png')
    cv2.imwrite(free_space_map_path, (-free_space_map + 1) * 255)

    return free_space_map, free_space_map_path


def generate_start_and_goal(points):
    while True:
        start_idx = np.random.choice(points.shape[0], 1)
        goal_idx = np.random.choice(points.shape[0], 1)
        start = np.squeeze(points[start_idx, :])
        goal = np.squeeze(points[goal_idx, :])
        if np.linalg.norm(start - goal) > min_distance:
            return start, goal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('map', type=str, help='map path. everything is created in the same dir')
    parser.add_argument('--ant_size', type=int, default=26,
                        help='Size of ant square for free space map.'
                             ' Ant can be ant_size/2 from wall (in maze pixels)')
    parser.add_argument('--num_workspaces', type=int, default=500,
                        help='number of workspaces to generate')
    parser.add_argument('--num_validation_workspaces', type=int, default=100,
                        help='number of validation workspaces to generate')
    parser.add_argument('--num_test_workspaces', type=int, default=100,
                        help='number of test workspaces to generate')
    parser.add_argument('--min_distance', type=int, default=20,
                        help='minimum distance between a target and a goal in maze pixels')

    args = parser.parse_args()
    num_of_workspaces = args.num_workspaces
    num_of_validation_workspaces = args.num_validation_workspaces
    num_of_test_workspaces = args.num_test_workspaces
    min_distance = args.min_distance
    workspaces_dir_name = os.path.dirname(args.map)



    free_space_map, free_space_map_path = prepeare_freespace(args.map, args.ant_size)

    start_goal_free_space_map = np.copy(free_space_map)
    free_space_points = np.argwhere(free_space_map == 0)
    # points where more than 2 neighbors are not free space are not valid for start or goal remove them
    for p in free_space_points:
        neighbors = [(p[0] + i, p[1] + j) for i in range(-1, 2) for j in range(-1, 2)]
        n_obstacle_neighbors = np.sum([(free_space_map[n] > 0) for n in neighbors])
        if n_obstacle_neighbors > 2:
            start_goal_free_space_map[p[0], p[1]] = 1
    # recompute:
    free_space_points = np.argwhere(start_goal_free_space_map == 0)

    workspaces = []
    validation_workspaces = []
    test_workspaces = []

    for i in range(num_of_workspaces):
        workspaces.append(generate_start_and_goal(free_space_points))
    for i in range(num_of_validation_workspaces):
        validation_workspaces.append(generate_start_and_goal(free_space_points))
    for i in range(num_of_test_workspaces):
        test_workspaces.append(generate_start_and_goal(free_space_points))

    # save workspaces (train, validation and test) to csv:
    workspaces = np.array(workspaces)
    np.save(workspaces_dir_name + '/workspaces', workspaces)
    validation_workspaces = np.array(validation_workspaces)
    np.save(workspaces_dir_name + '/validation_workspaces', validation_workspaces)
    test_workspaces = np.array(test_workspaces)
    np.save(workspaces_dir_name + '/test_workspaces', test_workspaces)


    # create plot for each train workspace and save
    Path(workspaces_dir_name + '/train_workspaces_plot').mkdir(parents=True, exist_ok=True)
    for i, w in enumerate(workspaces):
        w_im = np.copy(-free_space_map + 1)
        cv2.circle(w_im, tuple(reversed(w[0])), 1, 0.2, -1)
        cv2.circle(w_im, tuple(reversed(w[1])), 2, 0.4, -1)

        cv2.imwrite(workspaces_dir_name + f'/train_workspaces_plot/{i}.png', w_im * 255)

    # create plot of all train workspaces (in one plot)
    colors = sns.color_palette(None, num_of_workspaces)
    color_freespace_map = cv2.imread(free_space_map_path, cv2.IMREAD_COLOR) / 255
    # color_freespace_map = cv2.rotate(color_freespace_map, cv2.cv2.ROTATE_90_CLOCKWISE)
    for i, w in enumerate(workspaces):
        color_freespace_map[tuple(w[0])] = color_freespace_map[tuple(w[1])] = colors[i]

    cv2.imwrite(workspaces_dir_name + f'/train_workspaces_plot/all.png', color_freespace_map * 255)

