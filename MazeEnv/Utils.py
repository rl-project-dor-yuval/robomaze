import numpy as np


def make_circular_map(size, radius):
    """
    :param size : size of the map (has to be in class mz.MazeSize)
    :param radius : radius of the maze

    :return : bitmap numpy array of the maze to push to MazeEnv
    """
    center = np.divide(size, 2)
    x, y = np.ogrid[:size[0], :size[1]]
    maze_map = np.where(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) > radius, 1, 0)

    return maze_map
