from typing import Optional
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import pybullet_data
import os


class MazeSize:
    SMALL = (5, 10)
    MEDIUM = (15, 15)
    LARGE = (20, 20)


class Rewards:
    def __init__(self, target_arrival=1, collision=-1, timeout=0):
        """
        The collection of rewards and their values
        :param target_arrival: the reward's value for arriving the target
        :param collision: the reward's value for a collision
        :param timeout: the reward's value for timeout
        """
        self.target_arrival = target_arrival
        self.collision = collision
        self.timeout = timeout
        # TODO add more


class ObservationsDefinition:
    observations_opts = {"joint_state", "robot_loc", "robot_target_loc"}

    def __init__(self, observations: list = ["joint_state", "robot_loc", "robot_target_loc"]):
        for ob in observations:
            if ob not in self.observations_opts:
                raise ValueError

        self.observations = observations


class MazeEnv(gym.Env):
    def __init__(self, maze_size=MazeSize.MEDIUM , start_state=None, rewards: Rewards=None,
                 timeout_steps: int = 0, observations: ObservationsDefinition = None,):
        """
        :param maze_size: the size of the maze from : {MazeSize.SMALL, MazeSize.MEDIUM, MazeSize.LARGE}
        :param start_state: TODO: will include staff like start and end position,
        :param rewards: definition of reward values for events
        :param timeout_steps: maximum steps until getting timeout reward
         (if a timeout reward is defined)
        :param observations: definition of the desired observations for the agent
        :return: Maze Environment object

        Initializing environment object
        """
        # TODO handle default for all parameters
        # TODO validate maze size
        self.maze_size = maze_size
        #TODO validate start_state
        self.start_state = start_state

        self.rewards = rewards
        self.timeout_steps = timeout_steps

        if observations is None:
            # default observations:
            self.observations = ObservationsDefinition()
        else:
            self.observations = observations

    def step(self, action):

        p.stepSimulation()
        # TODO return observation, reward, is_done, info

    def reset(self, create_video=False):
        """
        reset the environment for the next episode
        :param create_video: weather to create video file from the next episode
        """
        # TODO Initialize pybullet simulation and maze
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0, 0, -10)

        floorUid = p.loadURDF("plane_transparent.urdf")
        p.changeVisualShape(objectUniqueId=floorUid,
                            linkIndex=-1,
                            rgbaColor=[0, 0.5, 0.7, 1])

        # load maze, TODO change to dynamic maze loading:
        # for i in range(-7, 8):
        #     cubeUid = p.loadURDF("cube.urdf", basePosition=[i, 7, 0.5])
        #     cubeUid = p.loadURDF("cube.urdf", basePosition=[i, -7, 0.5])
        #     cubeUid = p.loadURDF("cube.urdf", basePosition=[-7, i, 0.5])
        #     cubeUid = p.loadURDF("cube.urdf", basePosition=[7, i, 0.5])

        # load ant,
        self.antUid = p.loadMJCF("data/myAnt.xml")[0]
        # for i in range(-1,20):
        #     p.changeVisualShape(self.antUid, i, rgbaColor=(0.3,0.3,0.3,0.9))

        # sUid = p.loadURDF("data/s.urdf")

    def render(self):
        # TODO think if it is necessary
        pass




