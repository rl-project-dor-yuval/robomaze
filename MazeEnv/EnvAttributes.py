from dataclasses import dataclass
import numpy as np


class MazeSize:
    """
    3 different sizes that could be set for the maze
    """
    SQUARE5 = (5, 5)
    SQUARE10 = (10, 10)
    SQUARE15 = (15, 15)
    SQUARE20 = (20, 20)
    SMALL = (5, 10)
    MEDIUM = (10, 15)
    LARGE = SQUARE20


@dataclass
class Rewards:
    def __init__(self, target_arrival=1,
                 collision=-1,
                 timeout=0,
                 idle=0,
                 fall=-1,
                 target_distance_offset=0,
                 target_distance_weight=0,
                 rotation_weight=0,):
        """
        :param target_arrival: the reward's value for arriving the target
        :param collision: the reward's value for a collision
        :param timeout: the reward's value for timeout
        :param idle: the reward for a time step where nothing else happens
        :param fall: the reward for falling
        :param target_distance_offset: offset for target distance reward, should be around max target distance
            to promise negative reward
        :param target_distance_weight: the weight for the target distance reward.

        The collection of rewards and their values
        """
        self.idle = idle
        self.target_arrival = target_arrival
        self.collision = collision
        self.timeout = timeout
        self.fall = fall
        self.target_distance_offset = target_distance_offset
        self.target_distance_weight = target_distance_weight
        self.rotation_weight = rotation_weight

    def compute_target_distance_reward(self, target_distance):
        return self.target_distance_weight * (self.target_distance_offset - target_distance)

    def compute_rotation_reward(self, rotation_diff):
        return self.rotation_weight * (np.pi - np.abs(rotation_diff))

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(**loader.construct_mapping(node))
