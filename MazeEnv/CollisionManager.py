import numpy as np
import pybullet as p
from pybullet_utils import bullet_client as bc


class CollisionManager:
    target_uid: int
    maze_uids: np.ndarray
    ant_uid: int
    _pclient: bc.BulletClient

    def __init__(self, pybullet_client, maze_uids, target_uid, ant_uid, floor_uid):
        self._pclient = pybullet_client
        self.target_uid = target_uid
        self.maze_uids = maze_uids
        self.ant_uid = ant_uid
        self.floor_uid = floor_uid

        # disable collision for target sphere since we want it to be
        # "transparent for collisions" (ant can get inside)
        self._pclient.setCollisionFilterGroupMask(target_uid, -1, 0, 0)

    def check_hit_maze(self) -> bool:
        """
        checks if there were collision between the ant and the maze in the last step
        """
        contact_points = self._pclient.getContactPoints(bodyA=self.ant_uid)
        # each point is a tuple, the object the ant collided with is in
        # place 2 in the tuple
        for point in contact_points:
            if point[2] in self.maze_uids:
                return True

        return False

    def check_ant_target_collision(self) -> bool:
        """
        checks if there were a collision between the ant and  the target in the last step.
        this method was seperated from check maze collision since we don't use ant-target
        collision anymore for reward, but we want to preserve this code.
        """
        # since target sphere is masked for collision, we need to use
        # getClosestPoints to know if the ant is inside the sphere
        return self._pclient.getClosestPoints(self.ant_uid, self.target_uid, distance=0)

    def check_hit_floor(self):
        """
        checks if the torso (sphere of the body) of the ant hit the floor
        :return: hit_floor
        """
        # for some reason, the torso is linkIndex -1
        contact_points = self._pclient.getContactPoints(bodyA=self.ant_uid, linkIndexA=-1)
        for point in contact_points:
            if point[2] == self.floor_uid:
                return True

        return False
