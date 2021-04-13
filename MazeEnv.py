from typing import Optional

import gym
from gym import error, spaces, utils
from gym.utils import seeding

# change to named tuple or class
MAZE_SIZE_SMALL = (5, 10)
MAZE_SIZE_MEDIUM = (15, 15)
MAZE_SIZE_LARGE = (20, 20)


class Rewards:
    def __init__(self, target_arrival=1, collision=-1, timeout=0):
        """
        The collection of rewards that that you are intrested with in the upcoming simulation
        :param target_arrival: the reward's value for arriving the target
        :param collision: the reward's value for a collision
        :param timeout: the reward's value for timeout
        """
        self.target_arrival = target_arrival
        self.collision = collision
        self.timeout = timeout
        # TODO add more


class ObsDef:
    def __init__(self, observations: list = ["joint_state", "robot_loc", "robot_target"]):
        for ob in observations:
            if ob not in {"joint_state", "robot_loc", "robot_target"}:
                raise ValueError


class MazeEnv(gym.Env):
    def __init__(self, maze_size, start_state, rewards: Rewards,
                 timeout_steps: int = 0, observations: list = ["joint_state", "robot_loc", "robot_target"]):
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
