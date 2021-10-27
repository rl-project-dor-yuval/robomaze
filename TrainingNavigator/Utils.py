import numpy as np


def cart2pol(vec):
    r = np.linalg.norm(vec)
    theta = np.arctan2(vec[1], vec[0])
    return np.array([r, theta])


def pol2cart(vec):
    x = vec[0] * np.cos(vec[1])
    y = vec[0] * np.sin(vec[1])
    return np.array([x, y])