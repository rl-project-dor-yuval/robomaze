# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import time
from typing import Tuple, Dict, List

from TrainingNavigator.RRT_star.src.rrt.rrt_star import RRTStar
from TrainingNavigator.RRT_star.src.search_space.search_space import ImgSearchSpace
from TrainingNavigator.RRT_star.src.utilities.plotting import Plot
from Utils import plot_trajectory, blackwhiteswitch

# noinspection PyUnreachableCode
class TrajGenerator:
    """
    an object that generates trajectories using RRTstar
    """

    def __init__(self, mapPath: str, max_section_len=20):

        self.map = blackwhiteswitch(mapPath)
        self.map = cv2.rotate(self.map, cv2.cv2.ROTATE_90_CLOCKWISE)

        self.max_section_len = max_section_len

        X_dimensions = np.array([[0, self.map.shape[0] - 1], [0, self.map.shape[1] - 1]])  # dimensions of Search Space
        self.X = ImgSearchSpace(dimension_lengths=X_dimensions, O=None, Im=self.map)

        self.Q = np.array([(8, 4)])  # length of tree edges
        self.r = 1  # length of smallest edge to check for intersection with obstacles
        self.max_samples = 10**6  # max number of samples to take before timing out
        self.rewire_count = 32  # optional, number of nearby branches to rewire
        self.prc = 0.1  # probability of checking for a connection to goal

    # noinspection PyUnreachableCode
    def find_optimal_trajectories(self,
                                  xInit: Tuple[int, int],
                                  xGoal: Tuple[int, int],
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

            print(f"RRT* {i + 1}/{numOfTrajs} Time {end_t - start_t} [sec]")

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

                plot.plot_start(self.X, x_init)
                plot.plot_goal(self.X, x_goal)
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


if __name__ == "__main__":
    workspaces_file_path = "workspaces/Spiral15x15.npy"  # path of numpy file with workspaces
    workspaces_filename = workspaces_file_path.split("/")[-1].split(".")[0]

    trajs_filename = "Spiral15x15"
    plots_save_path = "workspaces/Spiral15x15_plots/"  # path to save plots
    Path(plots_save_path).mkdir(parents=True, exist_ok=True)

    # create Search Space
    map_path = "maps/Spiral15x15_freespace.png"

    # create Search Space
    maze_map = -(cv2.imread(map_path, cv2.IMREAD_GRAYSCALE) / 255) + 1
    map_granularity = 0.1  # in simulation coordinates, which means that any pixel in the map
    # is map_granularity units in the simulation coordinates

    np.set_printoptions(precision=1)

    trajGen = TrajGenerator(map_path, max_section_len=15)

    ws_list = np.load(workspaces_file_path)
    num_workspaces = ws_list.shape[0]
    ws_traj_dict = {}  # workspaces Trajectory Dictionary key = workspace idx, value= dictionary of

    for i in range(num_workspaces):
        # coords when origin is bottom left (this is how RRTstar algo works)
        x_init = (ws_list[i, 0, 1], trajGen.map.shape[0] - ws_list[i, 0, 0] - 1)
        x_goal = (ws_list[i, 1, 1], trajGen.map.shape[0] - ws_list[i, 1, 0] - 1)

        traj = trajGen.find_optimal_trajectories(xInit=x_init, xGoal=x_goal, numOfTrajs=1, plot=False)
        traj[0] = np.array(traj[0])
        # anyway plot manually:
        plot_trajectory(traj[0], maze_map, save_loc=plots_save_path + str(i) + "._traj_plot.png")

        traj[0] = traj[0] * map_granularity
        # TODO: meanwhile saving only the first Traj for each workspace
        ws_traj_dict[str(i)] = np.array(traj[0])

        np.set_printoptions(precision=1)
        print(i, '\n', ws_traj_dict[str(i)])

    np.savez(f'workspaces/{trajs_filename}_trajectories.npz', **ws_traj_dict)

    # test loading
    trajectories = np.load(f'workspaces/{trajs_filename}_trajectories.npz')
    assert np.all(trajectories[str(num_workspaces - 1)] == traj[0])
