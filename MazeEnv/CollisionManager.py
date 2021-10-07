import numpy as np
import pybullet as p
from pybullet_utils import bullet_client as bc

class CollisionManager:
    target_uid: int
    maze_uids: np.ndarray
    ant_uid: int
    _pclient: bc.BulletClient

    def __init__(self, pybullet_client, maze_uids, target_uid, ant_uid):
        self._pclient = pybullet_client
        self.target_uid = target_uid
        self.maze_uids = maze_uids
        self.ant_uid = ant_uid

        # disable collision for target sphere since we want it to be
        # "transparent for collisions" (ant can get inside)
        self._pclient.setCollisionFilterGroupMask(target_uid, -1, 0, 0)

    def check_ant_collisions(self):
        """
        checks if there were collision between the ant and the maze or target
        in the last pubullet.step() call

        :return: hit_target, hit_maze
        """
        hit_target = False
        hit_maze = False

        contact_points = self._pclient.getContactPoints(bodyA=self.ant_uid)
        # each point is a tuple, the object the ant collided with is in
        # place 2 in the tuple

        for point in contact_points:
            if point[2] in self.maze_uids:
                hit_maze = True
                break

        # since target sphere is masked for collision, we need to use
        # getClosestPoints to know if the ant is inside the sphere
        if self._pclient.getClosestPoints(self.ant_uid, self.target_uid, distance=0):
            hit_target = True

        return hit_target, hit_maze
