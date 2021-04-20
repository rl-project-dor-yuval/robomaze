import pybullet as p
import numpy as np
from os import path
import math


class Maze:
    _BLOCK_Z_COORD = 0.5  # half of block size so they won't be inside the floor

    def __init__(self, maze_size, block_map, target_position3d):
        self.maze_size = maze_size
        self.maze_frame_uids = np.zeros([4])

        self.floorUid = p.loadURDF("floor.urdf")
        self._load_maze_edges()
        self.target_sphereUid = p.loadURDF("data/goalSphere.urdf",
                                           basePosition=target_position3d)

    def _load_maze_edges(self):
        """load the blocks for the edges of the maze"""
        block_x_path = "data/block" + str(self.maze_size[0]) + ".urdf"
        block_y_path = "data/block" + str(self.maze_size[1]) + ".urdf"

        if not (path.exists(block_x_path) and path.exists(block_y_path)):
            raise Exception("Could not load maze at the given size,"
                            " no matching edges block were found."
                            " please use MazeSize.<desired size>")

        # along y blocks:
        self.maze_frame_uids[0] = p.loadURDF(block_y_path,
                                             basePosition=[-0.5,
                                                           self.maze_size[1] / 2,
                                                           self._BLOCK_Z_COORD])
        self.maze_frame_uids[1] = p.loadURDF(block_y_path,
                                             basePosition=[self.maze_size[0] + 0.5,
                                                           self.maze_size[1] / 2,
                                                           self._BLOCK_Z_COORD])

        # along x blocks:
        x_orientation = p.getQuaternionFromEuler([0, 0, math.pi / 2])
        self.maze_frame_uids[2] = p.loadURDF(block_x_path,
                                             basePosition=[self.maze_size[0] / 2,
                                                           -0.5,
                                                           self._BLOCK_Z_COORD],
                                             baseOrientation=x_orientation)
        self.maze_frame_uids[3] = p.loadURDF(block_x_path,
                                             basePosition=[self.maze_size[0] / 2,
                                                           self.maze_size[1] + 0.5,
                                                           self._BLOCK_Z_COORD],
                                             baseOrientation=x_orientation)