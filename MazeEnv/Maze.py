import pybullet
from pybullet_utils import bullet_client as bc
import numpy as np
from os import path
import math
from MazeEnv.EnvAttributes import MazeSize

_BLOCK_Z_COORD = 1  # so half of it won't be in the ground


class Maze:
    pubullet_client: bc.BulletClient
    maze_size: MazeSize
    maze_map: np.ndarray
    target_position3d: np.ndarray

    def __init__(self,
                 pybullet_client,
                 maze_size,
                 maze_map,
                 tile_size,
                 target_position3d,
                 target_heading,
                 optimize_boarders=True):
        """
        :param pybullet_client:
        :param maze_size: MazeSize object defining the maze size
        :param maze_map: numpy array describing the Maze
        :param tile_size: size of building block of the Maze
        :param target_position3d: 3d position of Goal
        :param optimize_boarders: if True,
            collision detection is checked only on boarder of free areas on the map
        :return Maze object
        """
        self.maze_size = maze_size

        self._pclient = pybullet_client

        self._floorUid = self._pclient.loadURDF("floor.urdf")
        self._pclient.changeDynamics(self._floorUid, -1, lateralFriction=10)

        self._maze_frame_uids = np.zeros([4])
        self._maze_frame_corners_uids = np.zeros([4])
        self._load_maze_edges()

        self._target_sphereUid = self._pclient.loadURDF("goalSphere.urdf",
                                                        basePosition=target_position3d)

        # setup direction pointer (disable collision and change color)
        pointer_orientation = self._pclient.getQuaternionFromEuler((0, 0, target_heading))
        pointer_position = list(target_position3d)
        pointer_position[2] += 0.75
        self._direction_pointer = self._pclient.loadURDF("direction_pointer.urdf",
                                                         basePosition=pointer_position,
                                                         baseOrientation=pointer_orientation,
                                                         globalScaling=1.25)
        self._pclient.setCollisionFilterGroupMask(self._direction_pointer, -1, 0, 0)  # disable collisions
        self._pclient.changeVisualShape(self._direction_pointer, linkIndex=-1, rgbaColor=[0, 0, 0.3, 0.5])

        # setup subgoal marker as well as direction pointer for it. It is invisible at the beginning:
        self._subgoal_marker = self._pclient.loadURDF("goalSphere.urdf",
                                                      basePosition=(0, 0, 0),
                                                      globalScaling=0.5)
        self._pclient.changeVisualShape(self._subgoal_marker, -1, rgbaColor=[0, 0, 0, 0])
        self._pclient.setCollisionFilterGroupMask(self._subgoal_marker, -1, 0, 0)

        self._subgoal_pointer = self._pclient.loadURDF("direction_pointer.urdf",
                                                       basePosition=(0, 0, 0),
                                                       globalScaling=1)
        self._pclient.setCollisionFilterGroupMask(self._subgoal_pointer, -1, 0, 0)
        self._pclient.changeVisualShape(self._subgoal_pointer, linkIndex=-1, rgbaColor=[0, 0, 0, 0])

        if optimize_boarders:
            maze_map_boarders, maze_map_fill = self._get_maze_map_boarders(maze_map)
            self._create_maze_urdf(maze_map_boarders, "curr_maze_boarders.urdf", tile_size)
            self._create_maze_urdf(maze_map_fill, "curr_maze_fill.urdf", tile_size)
            self._maze_tiles_uid = self._pclient.loadURDF("curr_maze_boarders.urdf")
            _ = self._pclient.loadURDF("curr_maze_fill.urdf")
        else:
            self._create_maze_urdf(maze_map, "curr_maze.urdf", tile_size)
            self._maze_tiles_uid = self._pclient.loadURDF("curr_maze.urdf")

    def get_maze_objects_uids(self):
        maze_uids = np.concatenate([self._maze_frame_uids,
                                    self._maze_frame_corners_uids,
                                    [self._maze_tiles_uid]])

        return maze_uids, self._target_sphereUid, self._floorUid

    def set_new_goal(self, target_position3d, target_heading):
        _, old_orientation = self._pclient.getBasePositionAndOrientation(self._target_sphereUid)
        self._pclient.resetBasePositionAndOrientation(self._target_sphereUid, target_position3d, old_orientation)

        pointer_position = list(target_position3d)
        pointer_position[2] += 0.75
        pointer_orientation = self._pclient.getQuaternionFromEuler((0, 0, target_heading))
        self._pclient.resetBasePositionAndOrientation(self._direction_pointer, pointer_position, pointer_orientation)

    def set_subgoal_marker(self, position=(0, 0), heading=0, show_direction_pointer=True, visible=True):
        """
        put a marker on the given position
        :param position: the position of the marker
        :param heading: heading of the pointer above the marker
        :param visible: set to false in order to remove the marker
        """
        if visible:
            position = (*position, 0)
            self._pclient.changeVisualShape(self._subgoal_marker, -1, rgbaColor=[0.5, 0.5, 0.5, 0.75])
            self._pclient.resetBasePositionAndOrientation(self._subgoal_marker, position, [0, 0, 0, 1])

            pointer_position = list(position)
            pointer_position[2] += 0.75
            pointer_orientation = self._pclient.getQuaternionFromEuler((0, 0, heading))
            self._pclient.changeVisualShape(self._subgoal_pointer, -1, rgbaColor=[0.2, 0.2, 0.2, 0.75])
            self._pclient.resetBasePositionAndOrientation(self._subgoal_pointer, pointer_position, pointer_orientation)
        else:
            self._pclient.changeVisualShape(self._subgoal_marker, -1, rgbaColor=[0, 0, 0, 0])
            self._pclient.changeVisualShape(self._subgoal_pointer, -1, rgbaColor=[0, 0, 0, 0])

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
                        '        <box size="{tile_size} {tile_size} 2" />\n'
                        '        </geometry>\n'
                        '      <material name="Cyan">\n'
                        '        <color rgba="0 1.0 1.0 1.0"/>\n'
                        '      </material>\n'
                        '    </visual>\n'
                        '    <collision>\n'
                        '      <origin xyz="{x} {y} {z}" rpy="0 0 0"/>\n'
                        '      <geometry>\n'
                        '        <box size="{tile_size} {tile_size} 2" />\n'
                        '      </geometry>\n'
                        '    </collision>\n'.format(x=x, y=y, z=_BLOCK_Z_COORD, tile_size=tile_size))

        f.write('    </link>\n</robot>\n')

        f.close()

    def _get_maze_map_boarders(self, maze_map):
        boarders = maze_map.copy()
        fill = np.zeros_like(maze_map)

        for i in range(1, maze_map.shape[0] - 1):
            for j in range(1, maze_map.shape[1] - 1):
                if maze_map[i, j] == 1:
                    if maze_map[i, j - 1] == 0 or \
                            maze_map[i, j + 1] == 0 or \
                            maze_map[i - 1, j] == 0 or \
                            maze_map[i + 1, j] == 0 or \
                            maze_map[i - 1, j - 1] == 0 or \
                            maze_map[i - 1, j + 1] == 0 or \
                            maze_map[i + 1, j - 1] == 0 or \
                            maze_map[i + 1, j + 1] == 0:
                        continue  # it is a boarder
                    else:
                        # not a boarder
                        boarders[i, j] = 0
                        fill[i, j] = 1

        return boarders, fill
