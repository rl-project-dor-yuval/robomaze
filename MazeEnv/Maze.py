import pybullet
from pybullet_utils import bullet_client as bc
import numpy as np
from os import path
import math


_BLOCK_Z_COORD = 0.5  # half of block size so they won't be inside the floor


class Maze:

    def __init__(self, pybullet_client, maze_size, maze_map, tile_size, target_position3d):
        from os import getcwd
        self.maze_size = maze_size

        self._pclient = pybullet_client
        self._floorUid = self._pclient.loadURDF("floor.urdf")
        self._maze_frame_uids = np.zeros([4])
        self._maze_frame_corners_uids = np.zeros([4])
        self._load_maze_edges()

        self._target_sphereUid = self._pclient.loadURDF("goalSphere.urdf",
                                                        basePosition=target_position3d)

        self._create_maze_urdf(maze_map, "curr_maze.urdf", tile_size)
        self._maze_tiles_uid = self._pclient.loadURDF("curr_maze.urdf")

    def get_maze_objects_uids(self):
        maze_uids = np.concatenate([self._maze_frame_uids,
                                    self._maze_frame_corners_uids,
                                    [self._maze_tiles_uid]])

        return maze_uids, self._target_sphereUid, self._floorUid

    def _load_maze_edges(self):
        """load the blocks for the edges of the maze"""
        block_x_path = "block" + str(self.maze_size[0]) + ".urdf"
        block_y_path = "block" + str(self.maze_size[1]) + ".urdf"

        # if not (path.exists(block_x_path) and path.exists(block_y_path)):
        #     raise Exception("Could not load maze at the given size,"
        #                     " no matching edges block were found."
        #                     " please use MazeSize.<desired size>")

        # along y blocks:
        self._maze_frame_uids[0] = self._pclient.loadURDF(block_y_path,
                                                          basePosition=[-0.5,
                                                                        self.maze_size[1] / 2,
                                                                        _BLOCK_Z_COORD])
        self._maze_frame_uids[1] = self._pclient.loadURDF(block_y_path,
                                                          basePosition=[self.maze_size[0] + 0.5,
                                                                        self.maze_size[1] / 2,
                                                                        _BLOCK_Z_COORD])

        # along x blocks:
        x_orientation = self._pclient.getQuaternionFromEuler([0, 0, math.pi / 2])
        self._maze_frame_uids[2] = self._pclient.loadURDF(block_x_path,
                                                          basePosition=[self.maze_size[0] / 2,
                                                                        -0.5,
                                                                        _BLOCK_Z_COORD],
                                                          baseOrientation=x_orientation)
        self._maze_frame_uids[3] = self._pclient.loadURDF(block_x_path,
                                                          basePosition=[self.maze_size[0] / 2,
                                                                        self.maze_size[1] + 0.5,
                                                                        _BLOCK_Z_COORD],
                                                          baseOrientation=x_orientation)

        # 4 corner blocks:
        self._maze_frame_corners_uids[0] = self._pclient.loadURDF("blockCube.urdf",
                                                                  basePosition=[-0.5,
                                                                                -0.5,
                                                                                _BLOCK_Z_COORD])
        self._maze_frame_corners_uids[1] = self._pclient.loadURDF("blockCube.urdf",
                                                                  basePosition=[self.maze_size[0] + 0.5,
                                                                                -0.5,
                                                                                _BLOCK_Z_COORD])
        self._maze_frame_corners_uids[2] = self._pclient.loadURDF("blockCube.urdf",
                                                                  basePosition=[-0.5,
                                                                                self.maze_size[1] + 0.5,
                                                                                _BLOCK_Z_COORD])
        self._maze_frame_corners_uids[3] = self._pclient.loadURDF("blockCube.urdf",
                                                                  basePosition=[self.maze_size[0] + 0.5,
                                                                                self.maze_size[1] + 0.5,
                                                                                _BLOCK_Z_COORD])

    @staticmethod
    def _create_maze_urdf(maze_grid, file_path, tile_size=0.1):
        """make one urdf file for the whole inisde of the maze."""
        f = open(file_path, "w+")
        f.write('<robot name="maze.urdf">\n'
                '  <static>true</static>\n'
                '  <link name="baselink">\n'
                '    <inertial>\n'
                '    <origin xyz="{o_x} {o_y} 0" rpy="0 0 0"/>\n'
                '    <mass value="0"/>\n'
                '    <inertia ixx="0"  ixy="0"  ixz="0" iyy="0" iyz="0" izz="0" />\n'
                '    </inertial>\n\n')  # .format(o_x=tile_size, o_y=tile_size))

        tiles_coord = tile_size * np.argwhere(maze_grid)
        tiles_coord += tile_size / 2
        if maze_grid is not None:
            for x, y in tiles_coord:
                f.write('    <visual>\n'
                        '      <origin xyz="{x} {y} {z}" rpy="0 0 0" />\n'
                        '        <geometry>\n'
                        '        <box size="{tile_size} {tile_size} 1" />\n'
                        '        </geometry>\n'
                        '      <material name="Cyan">\n'
                        '        <color rgba="0 1.0 1.0 1.0"/>\n'
                        '      </material>\n'
                        '    </visual>\n'
                        '    <collision>\n'
                        '      <origin xyz="{x} {y} {z}" rpy="0 0 0"/>\n'
                        '      <geometry>\n'
                        '        <box size="{tile_size} {tile_size} 1" />\n'
                        '      </geometry>\n'
                        '    </collision>\n'.format(x=x, y=y, z=_BLOCK_Z_COORD, tile_size=tile_size))

        f.write('    </link>\n</robot>\n')

        f.close()
