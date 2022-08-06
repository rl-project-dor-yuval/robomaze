import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import os


def create_workspaces(center, min_radius, max_radius, n_points=100):
    """
    :param center: The coordinates of the center of the Maze - where the radius is measured from
    :param min_radius: Minimum radius size from the center
    :param max_radius: Maximum radius size from the center
    :param n_points: number of points to be sampled

    :return: n_points numpy array of points in Maze coordinates and a heading for each point
    """
    angles = np.random.uniform(low=-math.pi, high=math.pi, size=(n_points,))
    radiuses = np.random.uniform(low=min_radius, high=max_radius, size=(n_points,))

    angles = np.expand_dims(angles, 1)
    radiuses = np.expand_dims(radiuses, 1)
    polar_points = np.concatenate((radiuses, angles), axis=1)

    points = np.array([[p[0] * math.cos(p[1]), p[0] * math.sin(p[1])] for p in polar_points])
    points += center
    return points


def plot_and_save_goals(workspaces, file_name):
    # plot train goals:
    fig, ax = plt.subplots()

    inner_circle = plt.Circle(center, min_radius, fill=False)
    outer_circle = plt.Circle(center, max_radius, fill=False)
    ax.add_artist(inner_circle)
    ax.add_artist(outer_circle)

    ax.scatter(workspaces[:, 0], workspaces[:, 1], label="Goals")
    ax.plot(center[0], center[1], 'ro', label="Initial Center of Robot")
    ax.margins(0.3)

    fig = plt.gcf()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "", file_name + "_plot.png"))
    plt.legend()
    plt.show()

    # Saving the Points Coordinates in CSV file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "", file_name + ".csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(workspaces)


base_filename = 'workspaces_06to3_'
center = np.array([5, 5])
min_radius = 0.6
max_radius = 3.0

n_train, n_val, n_test = 1000, 100, 200

train_workspaces = create_workspaces(center, min_radius, max_radius, n_train)
val_workspaces = create_workspaces(center, min_radius, max_radius, n_val)
test_workspaces = create_workspaces(center, min_radius, max_radius, n_test)

plot_and_save_goals(train_workspaces, base_filename + 'train')
plot_and_save_goals(val_workspaces, base_filename + 'validation')
plot_and_save_goals(test_workspaces, base_filename + 'test')
