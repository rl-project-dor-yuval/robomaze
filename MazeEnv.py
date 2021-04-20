from typing import Optional
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import pybullet_data
from os import path
import numpy as np
import math
from Recorder import Recorder


class MazeSize:
    """
    3 different sizes that could be set for the maze
    """
    SMALL = (5, 10)
    MEDIUM = (10, 15)
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


def start_state_is_valid(maze_size, start_state):
    """
    This function ensures that the locations are in the maze
    :param maze_size: tuple of the maze size (x,y)
    :param start_state: dictionary - {start_loc : tuple(3), target_loc : tuple(3)}
    """
    s_loc = start_state["start_loc"]
    t_loc = start_state["target_loc"]
    if s_loc[0] > maze_size[0] or s_loc[1] > maze_size[1] \
            or t_loc[0] > maze_size[0] or t_loc[1] > maze_size[1]:
        return False

    return True


class MazeEnv(gym.Env):

    _BLOCK_Z_COORD = 0.5  # half of block size so they won't be inside the floor
    _ANT_START_Z_COORD = 1  # the height the ant starts at
    zoom = 1.3  # is also relative to maze size
    default_rewards = Rewards()
    default_obs = ObservationsDefinition()

    recording_video_size = (800, 600)  # TODO make configurable (and maybe not static)
    recording_video_fps = 24

    physics_server = p.GUI  # TODO add setter?

    def __init__(self, maze_size=MazeSize.MEDIUM,
                 start_state: dict = {"start_loc": (1, 1, 0), "target_loc": (3, 3, 0)},
                 rewards: Rewards = default_rewards,
                 timeout_steps: int = 0,
                 observations: ObservationsDefinition = default_obs, ):
        """
        :param maze_size: the size of the maze from : {MazeSize.SMALL, MazeSize.MEDIUM, MazeSize.LARGE}
        :param start_state: dictionary - {start_loc : tuple(3), target_loc : tuple(3)}
        :param rewards: definition of reward values for events
        :param timeout_steps: maximum steps until getting timeout reward
         (if a timeout reward is defined)
        :param observations: definition of the desired observations for the agent
        :return: Maze Environment object

        Initializing environment object
        """
        self.recorder = Recorder()
        self.maze_frame_uids = np.zeros([4])
        self.antUid = None
        self.goal_sphereUid = None
        self.is_reset = False
        self.step_count = 0
        self.connectionUid = None
        self.episode_count = 0

        # TODO handle default for all parameters
        if not start_state_is_valid(maze_size, start_state):
            raise Exception("Start state is invalid")
        if timeout_steps < 0:
            raise Exception("timeout_steps value must be positive or zero for no limitation")

        self.maze_size = maze_size
        self.start_state = start_state
        self.rewards = rewards
        self.timeout_steps = timeout_steps

        if observations is None:
            # default observations:
            self.observations = ObservationsDefinition()
        else:
            self.observations = observations

    def step(self, action):
        if not self.is_reset:
            raise Exception("MazeEnv.reset() must be called before before MazeEnv.step()")


        p.stepSimulation()

        # >>do the step actions

        observation = self._get_observation()
        reward = self._get_reward()

        self.step_count += 1

        # if collision or exceeded time steps: is_done<-True

        if self.recorder.is_recording:
            # TODO maybe it is opposite and ..size[0] is for height
            _, _, im, _, _ = p.getCameraImage(width=self.recording_video_size[0],
                                              height=self.recording_video_size[1])
            self.recorder.insert_frame(im)

        # TODO return observation, reward, is_done, info

    def reset(self, create_video=False, reset_episode_count=False):
        """
        reset the environment for the next episode
        :param reset_episode_count: wather to reset the MazeEnv.episode_count value
        :param create_video: weather to create video file from the next episode
        """
        if self.connectionUid is not None:
            p.disconnect()

        self.connectionUid = p.connect(self.physics_server)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0, 0, -10)

        floorUid = p.loadURDF("floor.urdf")

        # load maze, TODO change to dynamic maze loading:
        self._load_maze_edges()

        # load ant, TODO change to start position
        self.antUid = p.loadMJCF("data/ant.xml")[0]
        p.resetBasePositionAndOrientation(self.antUid,
                                          [1, 1, 1],
                                          p.getBasePositionAndOrientation(self.antUid)[1])
        self._color_ant()

        # load goal sphere TODO change location to target location
        self.goal_sphereUid = p.loadURDF("data/goalSphere.urdf", basePosition=[2,2,0])

        # setup camera for a bird view
        p.resetDebugVisualizerCamera(cameraDistance=self.maze_size[1]/self.zoom,
                                     cameraYaw=0,
                                     cameraPitch=-89.9,
                                     cameraTargetPosition=[self.maze_size[0]/2, self.maze_size[1]/2, 0])

        if self.recorder.is_recording:
            self.recorder.save_recording_and_reset()
        if create_video:
            video_file_name = "episode" + str(self.episode_count) + ".avi"
            self.recorder.start_recording(video_file_name,
                                          self.recording_video_fps,
                                          self.recording_video_size)

        self.episode_count = 0 if reset_episode_count else self.episode_count
        self.episode_count += 1
        self.step_count = 0
        self.is_reset = True

    def render(self):
        # TODO think if it is necessary
        pass

    def _load_maze_edges(self):
        """load the blocks for the edges of the maze"""
        block_x_path = "data/block" + str(self.maze_size[0]) + ".urdf"
        block_y_path = "data/block" + str(self.maze_size[1]) + ".urdf"

        if not (path.exists(block_x_path) and path.exists(block_y_path)):
            raise Exception("Could not load maze at the given size,"
                            " no matching edges block were found."
                            " please use MazeSize.<desired size>")

        # along y blocks:
        self.maze_frame_uids[0] = p.loadURDF(block_y_path,
                                             basePosition=[-0.5,
                                                           self.maze_size[1]/2,
                                                           self._BLOCK_Z_COORD])
        self.maze_frame_uids[1] = p.loadURDF(block_y_path,
                                             basePosition=[self.maze_size[0] + 0.5,
                                                           self.maze_size[1]/2,
                                                           self._BLOCK_Z_COORD])

        # along x blocks:
        x_orientation = p.getQuaternionFromEuler([0, 0, math.pi/2])
        self.maze_frame_uids[2] = p.loadURDF(block_x_path,
                                             basePosition=[self.maze_size[0]/2,
                                                           -0.5,
                                                           self._BLOCK_Z_COORD],
                                             baseOrientation=x_orientation)
        self.maze_frame_uids[3] = p.loadURDF(block_x_path,
                                             basePosition=[self.maze_size[0]/2,
                                                           self.maze_size[1] + 0.5,
                                                           self._BLOCK_Z_COORD],
                                             baseOrientation=x_orientation)

    def _get_observation(self):
        pass

    def _get_reward(self):
        pass

    def _color_ant(self):
        pass



