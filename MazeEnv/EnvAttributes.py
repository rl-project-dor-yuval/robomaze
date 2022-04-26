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
    def __init__(self, target_arrival=1, collision=-1, timeout=0, idle=0, fall=-1):
        """
        :param target_arrival: the reward's value for arriving the target
        :param collision: the reward's value for a collision
        :param timeout: the reward's value for timeout
        :param idle: the reward for a time step where nothing else happens

        The collection of rewards and their values
        """
        self.idle = idle
        self.target_arrival = target_arrival
        self.collision = collision
        self.timeout = timeout
        self.fall = fall


@dataclass
class GoalCriteria:
    """
    definition of criteria to end episode with success.
    robot vertical velocity (sqrt(vx**2 + vy**2)) must be less than max_velocity
    both pitch and roll angle must be less than max_pitch_roll
    """
    max_velocity: float = 0.3
    max_pitch_roll: float = 0.6  # np.pi / 10

    def meets_criteria(self, Vx, Vy, pitch, roll):
        """
        :param Vx: Velocity on first axis
        :param Vy: Velocity on second axis
        :param pitch: Pitch angle
        :param roll: Roll angle
        :return: True if the criteria are met, False otherwise
        """
        return (np.sqrt(Vx ** 2 + Vy ** 2) < self.max_velocity) and \
               (np.abs(pitch) < self.max_pitch_roll) and\
               (np.abs(roll) < self.max_pitch_roll)

