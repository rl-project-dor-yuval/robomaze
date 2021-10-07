import math
import numpy as np
import matplotlib.pyplot as plt


def create_target_bank(center, min_radius, max_radius, n_points=100):
    angles = np.random.uniform(low=-math.pi, high=math.pi, size=(n_points,))
    radiuses = np.random.uniform(low=min_radius, high=max_radius, size=(n_points,))

    angles = np.expand_dims(angles, 1)
    radiuses = np.expand_dims(radiuses, 1)
    polar_points = np.concatenate((radiuses, angles), axis=1)

    points = np.array([[p[0]*math.cos(p[1]), p[0]*math.sin(p[1])] for p in polar_points])
    points += center

    return points


center = np.array([5, 5])
min_radius = 2
max_radius = 4
points = create_target_bank(center, min_radius, max_radius, 10)

out_circle = plt.Circle(center, max_radius,)
plt.gca().add_patch(out_circle)