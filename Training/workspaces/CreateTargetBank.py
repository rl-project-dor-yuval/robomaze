import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import os

def create_target_bank(center, min_radius, max_radius, target_heading_max_offset, n_points=100):
    """
    :param center: The coordinates of the center of the Maze - where the radius is measured from
    :param min_radius: Minimum radius size from the center
    :param max_radius: Maximum radius size from the center
    :param n_points: number of points to be sampled
    :param target_heading_max_offset: maximum offset for the target heading (in degrees)

    :return: n_points numpy array of points in Maze coordinates
    """
    angles = np.random.uniform(low=-math.pi, high=math.pi, size=(n_points,))
    radiuses = np.random.uniform(low=min_radius, high=max_radius, size=(n_points,))

    angles = np.expand_dims(angles, 1)
    radiuses = np.expand_dims(radiuses, 1)
    polar_points = np.concatenate((radiuses, angles), axis=1)

    points = np.array([[p[0] * math.cos(p[1]), p[0] * math.sin(p[1])] for p in polar_points])
    points += center

    headings = angles + np.random.uniform(low=-target_heading_max_offset,
                                          high=target_heading_max_offset,
                                          size=(n_points, 1))

    return np.concatenate((points, headings), axis=1)


def plot_and_save_goals(goals, file_name):
    # plot train goals:
    fig, ax = plt.subplots()

    inner_circle = plt.Circle(center, min_radius, fill=False)
    outer_circle = plt.Circle(center, max_radius, fill=False)
    ax.add_artist(inner_circle)
    ax.add_artist(outer_circle)

    u = np.cos(goals[:, 2])
    v = np.sin(goals[:, 2])

    ax.quiver(goals[:, 0], goals[:, 1], u, v, label="Goals With Direction")
    ax.plot(center[0], center[1], 'ro', label="Initial Center of Ant")
    ax.margins(0.3)

    fig = plt.gcf()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "", file_name + "_plot.png"))
    plt.legend()
    plt.show()

    # Saving the Points Coordinates in CSV file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "", file_name + ".csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(goals)


base_filename = 'goals_06to3_'
center = np.array([5, 5])
min_radius = 0.6
max_radius = 3.0
target_heading_max_offset = math.pi / 6

n_train, n_val, n_test = 1000, 200, 200

train_goals = create_target_bank(center, min_radius, max_radius, target_heading_max_offset, n_train)
val_goals = create_target_bank(center, min_radius, max_radius, target_heading_max_offset, n_val)
test_goals = create_target_bank(center, min_radius, max_radius, target_heading_max_offset, n_test)

plot_and_save_goals(train_goals, base_filename + 'train')
plot_and_save_goals(val_goals, base_filename + 'validation')
plot_and_save_goals(test_goals, base_filename + 'test')
