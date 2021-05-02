import pybullet as p
import numpy as np
from os import path
import math


_BLOCK_Z_COORD = 0.5  # half of block size so they won't be inside the floor


class Maze:

    def __init__(self, maze_size, block_map, target_position3d):
        self.maze_size = maze_size

        self._floorUid = p.loadURDF("floor.urdf")
        self._maze_frame_uids = np.zeros([4])
        self._maze_frame_corners_uids = np.zeros([4])
        self._load_maze_edges()

        self._target_sphereUid = p.loadURDF("data/goalSphere.urdf",
                                            basePosition=target_position3d)

    def get_maze_objects_uids(self):
        maze_uids = np.concatenate([self._maze_frame_uids, self._maze_frame_corners_uids])
        return maze_uids, self._target_sphereUid

    def _load_maze_edges(self):
        """load the blocks for the edges of the maze"""
        block_x_path = "data/block" + str(self.maze_size[0]) + ".urdf"
        block_y_path = "data/block" + str(self.maze_size[1]) + ".urdf"

        if not (path.exists(block_x_path) and path.exists(block_y_path)):
            raise Exception("Could not load maze at the given size,"
                            " no matching edges block were found."
                            " please use MazeSize.<desired size>")

        # along y blocks:
        self._maze_frame_uids[0] = p.loadURDF(block_y_path,
                                              basePosition=[-0.5,
                                                            self.maze_size[1] / 2,
                                                            _BLOCK_Z_COORD])
        self._maze_frame_uids[1] = p.loadURDF(block_y_path,
                                              basePosition=[self.maze_size[0] + 0.5,
                                                            self.maze_size[1] / 2,
                                                            _BLOCK_Z_COORD])

        # along x blocks:
        x_orientation = p.getQuaternionFromEuler([0, 0, math.pi / 2])
        self._maze_frame_uids[2] = p.loadURDF(block_x_path,
                                              basePosition=[self.maze_size[0] / 2,
                                                            -0.5,
                                                            _BLOCK_Z_COORD],
                                              baseOrientation=x_orientation)
        self._maze_frame_uids[3] = p.loadURDF(block_x_path,
                                              basePosition=[self.maze_size[0] / 2,
                                                            self.maze_size[1] + 0.5,
                                                            _BLOCK_Z_COORD],
                                              baseOrientation=x_orientation)

        # 4 corner blocks:
        self._maze_frame_corners_uids[0] = p.loadURDF("data/blockCube.urdf",
                                                      basePosition=[-0.5,
                                                                    -0.5,
                                                                    _BLOCK_Z_COORD])
        self._maze_frame_corners_uids[1] = p.loadURDF("data/blockCube.urdf",
                                                      basePosition=[self.maze_size[0] + 0.5,
                                                                    -0.5,
                                                                    _BLOCK_Z_COORD])
        self._maze_frame_corners_uids[2] = p.loadURDF("data/blockCube.urdf",
                                                      basePosition=[-0.5,
                                                                    self.maze_size[1] + 0.5,
                                                                    _BLOCK_Z_COORD])
        self._maze_frame_corners_uids[3] = p.loadURDF("data/blockCube.urdf",
                                                      basePosition=[self.maze_size[0] + 0.5,
                                                                    self.maze_size[1] + 0.5,
                                                                    _BLOCK_Z_COORD])
