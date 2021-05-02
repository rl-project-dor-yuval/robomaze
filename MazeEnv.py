from typing import Optional
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import pybullet_data
import numpy as np
import math
from Recorder import Recorder
from EnvAttributes import Rewards, ObservationsDefinition, MazeSize
from CollisionManager import CollisionManager
from Ant import Ant
from Maze import Maze


#TODO change "start_state" name? maybe seperate vars

def start_state_is_valid(maze_size, start_state):
    """
    This function ensures that the locations are in the maze
    :param maze_size: tuple of the maze size (x,y)
    :param start_state: dictionary - {start_loc : tuple(3), target_loc : tuple(3)}
    """
    s_loc = start_state["start_loc"]
    t_loc = start_state["target_loc"]
    # TODO - fix to make sure coordinates are >1. also make MazeEnv member function
    if s_loc[0] > maze_size[0] or s_loc[1] > maze_size[1] \
            or t_loc[0] > maze_size[0] or t_loc[1] > maze_size[1]:
        return False

    return True


class MazeEnv(gym.Env):

    _ANT_START_Z_COORD = 1  # the height the ant starts at
    recording_video_size = (800, 600)  # TODO make configurable (and maybe not static)
    zoom = 1.2  # is also relative to maze size

    physics_server = p.GUI  # TODO add setter?

    def __init__(self, maze_size=MazeSize.MEDIUM,
                 start_state: dict = {"start_loc": (1, 1, 0), "target_loc": (3, 3, 0)},
                 rewards: Rewards = Rewards(),
                 timeout_steps: int = 0,
                 observations: ObservationsDefinition = ObservationsDefinition(), ):
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

        # TODO handle default for all parameters
        if not start_state_is_valid(maze_size, start_state):
            raise Exception("Start state is invalid")
        if timeout_steps < 0:
            raise Exception("timeout_steps value must be positive or zero for no limitation")

        self.is_reset = False
        self.step_count = 0
        self.episode_count = 0
        self.start_state = start_state
        self.rewards = rewards
        self.timeout_steps = timeout_steps

        # setup simulation:
        self.connectionUid = p.connect(self.physics_server)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # load maze:
        # TODO handle passing the map instead of None
        self.maze = Maze(maze_size, None, start_state["target_loc"])

        # load ant robot:
        self.ant = Ant(self.start_state["start_loc"])

        # create collision detector and pass relevant uids:
        maze_uids, target_sphere_uid = self.maze.get_maze_objects_uids()
        self.collision_manager = CollisionManager(maze_uids,
                                                  target_sphere_uid,
                                                  self.ant.uid)

        # setup camera for a bird view:
        p.resetDebugVisualizerCamera(cameraDistance=self.maze.maze_size[1] / self.zoom,
                                     cameraYaw=0,
                                     cameraPitch=-89.9,
                                     cameraTargetPosition=[self.maze.maze_size[0] / 2, self.maze.maze_size[1] / 2, 0])

        self.recorder = Recorder(maze_size=maze_size,
                                 video_size=self.recording_video_size,
                                 zoom=self.zoom)

    def step(self, action):
        if not self.is_reset:
            raise Exception("MazeEnv.reset() must be called before before MazeEnv.step()")

        # pass actions through the ant object:
        # TODO dor implement:
        self.ant.action(action)

        # run simulation step
        p.stepSimulation()

        # check for ant collision in the last step:
        hit_target, hit_maze = self.collision_manager.check_ant_collisions()
        if hit_target:
            # TODO: update reward instead of printing
            print("hit target")
        if hit_maze:
            print("hit maze")

        observation = self._get_observation()
        reward = self._get_reward()

        self.step_count += 1

        # TODO if collision or exceeded time steps: is_done<-True

        if self.recorder.is_recording:
            self.recorder.insert_current_frame()

        # TODO return observation, reward, is_done, info

    def reset(self, create_video=False, reset_episode_count=False):
        """
        reset the environment for the next episode
        :param reset_episode_count: wather to reset the MazeEnv.episode_count value
        :param create_video: weather to create video file from the next episode
        """
        # move ant to start position:
        self.ant.reset()

        # handle recording (save last episode if needed)
        if self.recorder.is_recording:
            self.recorder.save_recording_and_reset()
        if create_video:
            video_file_name = "episode" + str(self.episode_count) + ".avi"
            self.recorder.start_recording(video_file_name)

        # update self state:
        self.episode_count = 0 if reset_episode_count else self.episode_count
        self.episode_count += 1
        self.step_count = 0
        self.is_reset = True

    def render(self):
        # TODO think if it is necessary
        pass

    def _get_observation(self):
        # if "joint_state" in ...
            # observation["joint_state "] =
        pass

    def _get_reward(self):
        pass



