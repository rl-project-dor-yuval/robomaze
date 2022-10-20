# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import sys
sys.path.append('.')
import argparse
import os.path
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import time
from typing import Tuple, Dict, List

from TrainingNavigator.RRT_star.src.rrt.rrt_star import RRTStar
from TrainingNavigator.RRT_star.src.search_space.search_space import ImgSearchSpace
from TrainingNavigator.RRT_star.src.utilities.plotting import Plot
from TrainingNavigator.Utils import plot_trajectory, blackwhiteswitch

# noinspection PyUnreachableCode
class TrajGenerator:
    """
    an object that generates trajectories using RRTstar
    """

    def __init__(self, mapPath: str, max_section_len=25):

        self.map = blackwhiteswitch(mapPath)
        self.map = cv2.rotate(self.map, cv2.cv2.ROTATE_90_CLOCKWISE)

        self.max_section_len = max_section_len

        X_dimensions = np.array([[0, self.map.shape[0] - 1], [0, self.map.shape[1] - 1]])  # dimensions of Search Space
        self.X = ImgSearchSpace(dimension_lengths=X_dimensions, O=None, Im=self.map)

        self.Q = np.array([(max_section_len, max_section_len//2, max_section_len//4, 1)])  # length of tree edges
        self.r = 1  # length of smallest edge to check for intersection with obstacles
        self.max_samples = 10**6 # max number of samples to take before timing out
        self.rewire_count = 16  # optional, number of nearby branches to rewire
        self.prc = 0.1  # probability of checking for a connection to goal

    # noinspection PyUnreachableCode
    def find_optimal_trajectories(self,
                                  xInit: Tuple[float, float],
                                  xGoal: Tuple[float, float],
                                  numOfTrajs: int,
                                  plot: bool) -> Dict[int, List[Tuple[int, int]]]:
        """
        This method generates a dictionary of optimal paths.
        :param xInit: initial location
        :param xGoal: Goal location
        :param numOfTrajs: number of iterations to run RRTstar (each run generates 1 optimal Traj)
        :param plot: plot trajectories and map flag
        return: dict(key= traj idx, value= list([(x1,y1),...(xn,yn)])
        """
        optimal_trajs = {}

        for i in range(numOfTrajs):

            start_t = time.time()
            rrt = RRTStar(self.X, self.Q, xInit, xGoal, self.max_samples, self.r, self.prc, self.rewire_count)
            path = rrt.rrt_star()
            end_t = time.time()

            # print(f"RRT* {i + 1}/{numOfTrajs} Time {end_t - start_t} [sec]")

            if path is not None:
                # convert trajectory to integers, and fixing the origin to be on
                # top left corner, since origin of algorithm is buttom left corner
                # plot is still done related to buttom left origin.
                path = self._cut_long_sections(path)
                optimal_trajs[i] = [(self.map.shape[0] - y - 1, x) for (x, y) in path]
            else:
                print(f"No path for Traj {i}")

            if plot:
                # plot
                PIL_maze_map = Image.open(map_path)
                plot = Plot("rrt_star_2d", PIL_maze_map)
                plot.plot_tree(self.X, rrt.trees)

                if path is not None:
                    plot.plot_path(self.X, path)
                else:
                    print("No Path was found")

                plot.plot_start(self.X, xInit)
                plot.plot_goal(self.X, xGoal)
                plot.draw(auto_open=True)

        return optimal_trajs

    def _cut_long_sections(self, trajectory):
        """
        given a trajectory, checks if there are sections shorter then self.max_section_len
        and replace them with multiple shorter sections
        """
        new_traj = [trajectory[0]]

        for i in range(len(trajectory) - 1):
            sec_distance = np.linalg.norm(np.array(trajectory[i]) - np.array(trajectory[i+1]))
            if sec_distance < self.max_section_len:
                new_traj.append(trajectory[i+1])
            else:
                n_subsections = int(np.ceil(sec_distance / self.max_section_len))
                # we take [1:0] because we dont want the first point, it is already the previous point
                new_points_x = np.linspace(trajectory[i][0], trajectory[i+1][0], n_subsections + 1)[1:].tolist()
                new_points_y = np.linspace(trajectory[i][1], trajectory[i+1][1], n_subsections + 1)[1:].tolist()
                new_traj.extend(list(zip(new_points_x, new_points_y)))

        return new_traj


def generate_traj_set(trajGen: TrajGenerator, ws_dir, maze_map, ws_list, filename, map_granularity):
    num_ws = ws_list.shape[0]

    plot_dir = os.path.join(ws_dir, f"{filename}_plots")
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    ws_traj_dict = {}  # workspaces Trajectory Dictionary key = workspace idx, value= dictionary of
    for i in range(num_ws):
        # coords when origin is bottom left (this is how RRTstar algo works)
        x_init = (ws_list[i, 0, 1], trajGen.map.shape[0] - ws_list[i, 0, 0] - 1)
        x_goal = (ws_list[i, 1, 1], trajGen.map.shape[0] - ws_list[i, 1, 0] - 1)

        traj = trajGen.find_optimal_trajectories(xInit=x_init, xGoal=x_goal, numOfTrajs=1, plot=False)
        traj[0] = np.array(traj[0])
        # anyway plot manually:
        plot_trajectory(traj[0], maze_map, save_loc=plot_dir + f'/{i}.png')

        traj[0] = traj[0] * map_granularity
        # TODO: meanwhile saving only the first Traj for each workspace
        ws_traj_dict[str(i)] = np.array(traj[0])

        np.set_printoptions(precision=1)
        print(i, '\n', ws_traj_dict[str(i)])

    np.savez(os.path.join(ws_dir, filename), **ws_traj_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('workspace_dir', type=str, help='everything is created in the same dir')
    parser.add_argument('--max_section_len', type=int, default=20,
                        help='maximum distance between two points in the trajectory')
    parser.add_argument('--map_granularity_inverse', type=float, default=10,
                        help='inverse of the map granularity (if map granularity is 1/3 then set to 3 for example)'
                             'use inverse to avoid floating point errors')
    parser.add_argument('--additional_name', type=str, default='',
                        help='additional name to be added to the file name')
    parser.add_argument('--freespace_map', action='store_true')
    parser.add_argument('--no_freespace_map', dest='freespace_map', action='store_false')
    parser.set_defaults(freespace_map=True)

    args = parser.parse_args()

    workspaces_train_path = args.workspace_dir + "/workspaces.npy"
    workspaces_test_path = args.workspace_dir + "/test_workspaces.npy"
    workspaces_validation_path = args.workspace_dir + "/validation_workspaces.npy"

    ws_train = np.load(workspaces_train_path)
    ws_test = np.load(workspaces_test_path)
    ws_validation = np.load(workspaces_validation_path)

    if args.freespace_map:
        map_path = args.workspace_dir + '/' + os.path.basename(os.path.normpath(args.workspace_dir)) + "_freespace.png"
        traj_file_name_train = "trajectories_train" + args.additional_name
        traj_file_name_test = "trajectories_test" + args.additional_name
        traj_file_name_validation = "trajectories_validation" + args.additional_name
    else:
        map_path = args.workspace_dir + '/' + os.path.basename(os.path.normpath(args.workspace_dir)) + ".png"
        traj_file_name_train = "trajectories_train_no_freespace" + args.additional_name
        traj_file_name_test = "trajectories_test_no_freespace" + args.additional_name
        traj_file_name_validation = "trajectories_validation_no_freespace" + args.additional_name

    maze_map = -(cv2.imread(map_path, cv2.IMREAD_GRAYSCALE) / 255) + 1

    np.set_printoptions(precision=1)

    trajGen = TrajGenerator(map_path, max_section_len=args.max_section_len)

    generate_traj_set(trajGen, args.workspace_dir, maze_map, ws_train, traj_file_name_train,
                      1./args.map_granularity_inverse)
    generate_traj_set(trajGen, args.workspace_dir, maze_map, ws_test, traj_file_name_test,
                      1./args.map_granularity_inverse)
    generate_traj_set(trajGen, args.workspace_dir, maze_map, ws_validation, traj_file_name_validation,
                      1./args.map_granularity_inverse)
