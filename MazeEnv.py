from typing import Optional
from enum import Enum

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class MazeSize:
    SMALL = (5, 10)
    MEDIUM = (15, 15)
    LARGE = (20, 20)


class Rewards:
    def __init__(self, target_arrival=1, collision=-1, timeout=0):
        self.target_arrival = target_arrival
        self.collision = collision
        self.timeout = timeout
        # TODO add more


class MazeEnv(gym.Env):

    def __init__(self, maze_size: tuple, start_state, rewards: Rewards,
                 timeout_steps: int = 0, observations: list = ["joint_state"]):
        """
        :param maze_size: TODO: define named tuple
        :param start_state: TODO: will include staff like start and end position,
        :param rewards: definition of reward values for events
        :param timeout_steps: maximum steps until getting timeout reward
         (if a timeout reward is defined)
        :param observations: definition of the desired observations for the agent
        :return: Maze Environment object

        Initializing environment object
        """
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass




