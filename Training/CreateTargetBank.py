import math
import csv
import numpy as np
import matplotlib.pyplot as plt


def create_target_bank(center, min_radius, max_radius, n_points=100):
    """
    :param center: The coordinates of the center of the Maze - where the radius is measured from
    :param min_radius: Minimum radius size from the center
    :param max_radius: Maximum radius size from the center
    :param n_points: number of points to be sampled

    :return: n_points numpy array of points in Maze coordinates
    """
    angles = np.random.uniform(low=-math.pi, high=math.pi, size=(n_points,))
    radiuses = np.random.uniform(low=min_radius, high=max_radius, size=(n_points,))

    angles = np.expand_dims(angles, 1)
    radiuses = np.expand_dims(radiuses, 1)
    polar_points = np.concatenate((radiuses, angles), axis=1)

    points = np.array([[p[0] * math.cos(p[1]), p[0] * math.sin(p[1])] for p in polar_points])
    points += center

    return points


center = np.array([5, 5])
min_radius = 2
max_radius = 4
points = create_target_bank(center, min_radius, max_radius, 20)

# Plotting and saving the test coords
fig, ax = plt.subplots()
ax.scatter(points[:, 0], points[:, 1], label="Targets")
ax.plot(center[0], center[1], 'ro', label="Initial Center of Ant")
fig = plt.gcf()
fig.savefig('./TestTargets/test_coords_plot.png')
plt.legend()
plt.show()

# Saving the Points Coordinates in CSV file
with open('./TestTargets/test_coords.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(points)
