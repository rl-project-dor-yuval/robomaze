# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import time

from TrainingNavigator.RRT.src.rrt.rrt_star import RRTStar
from TrainingNavigator.RRT.src.search_space.search_space import SearchSpace, ImgSearchSpace
from TrainingNavigator.RRT.src.utilities.plotting import Plot

x_init = (82, 80)  # starting location (0,0) is bottom left
x_goal = (15, 80)  # goal location (0,0) is bottom left

Q = np.array([(8, 4)])  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 2048  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.1  # probability of checking for a connection to goal

# create Search Space
map_path = "../../../../scratch/bottleneck_freespace.png"

maze_map = -(cv2.imread(map_path, cv2.IMREAD_GRAYSCALE) / 255) + 1
maze_map = cv2.rotate(maze_map, cv2.cv2.ROTATE_90_CLOCKWISE)
fig, ax1 = plt.subplots(1, 1)
ax1.imshow(-maze_map + 1, cmap='gray')

X_dimensions = np.array([[0, maze_map.shape[0] - 1], [0, maze_map.shape[1] - 1]])  # dimensions of Search Space
X = ImgSearchSpace(dimension_lengths=X_dimensions, O=None, Im=maze_map)

# create rrt_search
start_t = time.time()
rrt = RRTStar(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
path = rrt.rrt_star()
end_t = time.time()
print(f"RRT* Time {end_t - start_t} [sec]")

# plot
PIL_maze_map = Image.open(map_path)
plot = Plot("rrt_star_2d", PIL_maze_map)
plot.plot_tree(X, rrt.trees)

if path is not None:
    plot.plot_path(X, path)
else:
    print("No Path was found")

plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)
