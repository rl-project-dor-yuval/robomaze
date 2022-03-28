# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np
from PIL import Image
import csv
import matplotlib.pyplot as plt
import cv2
import time
from typing import Tuple, Dict, List

from TrainingNavigator.RRT_star.src.rrt.rrt_star import RRTStar
from TrainingNavigator.RRT_star.src.search_space.search_space import ImgSearchSpace
from TrainingNavigator.RRT_star.src.utilities.plotting import Plot


class TrajGenerator:
    """
    an object that generates trajectories using RRTstar
    """

    def __init__(self, mapPath: str):

        self.map = -(cv2.imread(mapPath, cv2.IMREAD_GRAYSCALE) / 255) + 1
        self.map = cv2.rotate(self.map, cv2.cv2.ROTATE_90_CLOCKWISE)

        X_dimensions = np.array([[0, self.map.shape[0] - 1], [0, self.map.shape[1] - 1]])  # dimensions of Search Space
        self.X = ImgSearchSpace(dimension_lengths=X_dimensions, O=None, Im=self.map)

        self.Q = np.array([(8, 4)])  # length of tree edges
        self.r = 1  # length of smallest edge to check for intersection with obstacles
        self.max_samples = 10**6  # max number of samples to take before timing out
        self.rewire_count = 32  # optional, number of nearby branches to rewire
        self.prc = 0.1  # probability of checking for a connection to goal

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
                optimal_trajs[i] = [(self.map.shape[0] - int(y), int(x)) for (x, y) in path]
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

        def traj_to_transitions():
            # TODO: implement a method that creates the trajectory's experiences to enter the RB
            pass


# create Search Space
map_path = "maps/bottleneck_freespace.png"

#
# fig, ax1 = plt.subplots(1, 1)
# ax1.imshow(-maze_map + 1, cmap='gray')


if __name__ == "__main__":
    trajGen = TrajGenerator(map_path)

    ws_list = np.load("workspaces/bottleneck.npy")
    num_workspaces = ws_list.shape[0]
    ws_traj_dict = {}  # workspaces Trajectory Dictionary key = workspace idx, value= dictionary of

    for i in range(num_workspaces):
        # coords when origin is bottom left (this is how RRTstar algo works)
        x_init = (ws_list[i, 0, 1], trajGen.map.shape[0] - ws_list[i, 0, 0] - 1)
        x_goal = (ws_list[i, 1, 1], trajGen.map.shape[0] - ws_list[i, 1, 0] - 1)

        traj = trajGen.find_optimal_trajectories(xInit=x_init, xGoal=x_goal, numOfTrajs=1, plot=False)
        # TODO: meanwhile saving only the first Traj for each workspace
        ws_traj_dict[str(i)] = np.array(traj[0])

        print(i, traj[0])

    np.savez('workspaces/botttleneck_trajectories.npz', **ws_traj_dict)

    # test loading
    trajectories = np.load('workspaces/botttleneck_trajectories.npz')
    assert np.all(trajectories[str(num_workspaces-1)] == traj[0])
