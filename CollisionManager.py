import numpy as np
import pybullet as p


class CollisionManager:
    target_uid: int
    maze_uids: np.ndarray
    ant_uid: int

    def __init__(self, maze_uids, target_uid, ant_uid):
        self.target_uid = target_uid
        self.maze_uids = maze_uids
        self.ant_uid = ant_uid

        # disable collision for target sphere since we want it to be
        # "transparent for collisions" (ant can get inside)
        p.setCollisionFilterGroupMask(target_uid, -1, 0, 0)

        print("ant:", ant_uid)
        print("maze:", maze_uids)
        print("target:", target_uid)

    def check_ant_collisions(self):
        """
        checks if there were collision between the ant and the maze or target
        in the last pubullet.step() call

        :return: hit_target, hit_maze
        """
        hit_target = False
        hit_maze = False

        contact_points = p.getContactPoints(bodyA=self.ant_uid)
        # each point is a tuple, the object the ant collided with is in
        # place 2 in the tuple

        for point in contact_points:
            if point[2] in self.maze_uids:
                hit_maze = True
                break

        # since target sphere is masked for collision, we need to use
        # getClosestPoints to know if the ant is inside the sphere
        if p.getClosestPoints(self.ant_uid, self.target_uid, distance=0):
            hit_target = True

        return hit_target, hit_maze
