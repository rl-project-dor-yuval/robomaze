from typing import Optional, Tuple
import gym
from gym import error, spaces, utils
from gym.spaces import Box
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


_ANT_START_Z_COORD = 1  # the height the ant starts at


class MazeEnv(gym.Env):

    # public members:
    rewards: Rewards

    timeout_steps: int
    step_count: int
    is_reset: bool
    episode_count: int

    recording_video_size: Tuple[int, int] = (800, 600)
    video_skip_frames: int = 4
    zoom: float = 1.2  # is also relative to maze size

    # private members:
    _collision_manager: CollisionManager
    _maze: Maze
    _ant: Ant
    _recorder: Recorder

    _start_loc: Tuple[float, float, float]
    _target_loc: Tuple[float, float, float]

    _physics_server = p.GUI  # TODO add setter?
    _connectionUid: int

    def __init__(self, maze_size=MazeSize.MEDIUM,
                 start_loc=(1, 1),
                 target_loc=(3, 3),
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
        if not self._start_state_is_valid(maze_size, start_loc, target_loc):
            raise Exception("Start state is invalid")
        if timeout_steps < 0:
            raise Exception("timeout_steps value must be positive or zero for no limitation")

        self.is_reset = False
        self.step_count = 0
        self.episode_count = 0
        self._start_loc = (start_loc[0], start_loc[1], _ANT_START_Z_COORD)
        self._target_loc = (target_loc[0], target_loc[1], 0)
        self.rewards = rewards
        self.timeout_steps = timeout_steps

        self.action_space = Box(low=-1, high=1, shape=(8,), dtype=np.float64)
        self.observation_space =

        # setup simulation:
        self._connectionUid = p.connect(self._physics_server)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # load maze:
        # TODO handle passing the map instead of None
        self._maze = Maze(maze_size, None, self._target_loc)

        # load ant robot:
        self._ant = Ant(self._start_loc)

        # create collision detector and pass relevant uids:
        maze_uids, target_sphere_uid = self._maze.get_maze_objects_uids()
        self._collision_manager = CollisionManager(maze_uids,
                                                   target_sphere_uid,
                                                   self._ant.uid)

        # setup camera for a bird view:
        p.resetDebugVisualizerCamera(cameraDistance=self._maze.maze_size[1] / self.zoom,
                                     cameraYaw=0,
                                     cameraPitch=-89.9,
                                     cameraTargetPosition=[self._maze.maze_size[0] / 2, self._maze.maze_size[1] / 2, 0])

        self._recorder = Recorder(maze_size=maze_size,
                                  video_size=self.recording_video_size,
                                  zoom=self.zoom)

    def step(self, action):
        if not self.is_reset:
            raise Exception("MazeEnv.reset() must be called before before MazeEnv.step()")
        assert self.action_space.contains(action), "Expected shape (8,) and value in [-1,1] "

        # initialize return values:
        observation = None  # TODO: handle
        reward = 0
        is_done = False
        info = None  # TODO: handle?

        # pass actions through the ant object:
        # TODO dor implement:
        self._ant.action(action)

        # run simulation step
        p.stepSimulation()

        # check for ant collision in the last step and update reward:
        hit_target, hit_maze = self._collision_manager.check_ant_collisions()
        if hit_target:
            reward += self.rewards.target_arrival
        if hit_maze:
            reward += self.rewards.collision
            is_done = True

        self.step_count += 1

        # check for timeout:
        if self.timeout_steps != 0 and self.step_count >= self.timeout_steps:
            reward += self.rewards.timeout
            is_done = True

        # handle recording
        if self._recorder.is_recording and \
                self.step_count % self.video_skip_frames == 0:
            self._recorder.insert_current_frame()

        return observation, reward, is_done, info

    def reset(self, create_video=False, reset_episode_count=False):
        """
        reset the environment for the next episode
        :param reset_episode_count: wather to reset the MazeEnv.episode_count value
        :param create_video: weather to create video file from the next episode
        """
        # move ant to start position:
        self._ant.reset()

        # handle recording (save last episode if needed)
        if self._recorder.is_recording:
            self._recorder.save_recording_and_reset()
        if create_video:
            video_file_name = "episode" + str(self.episode_count) + ".avi"
            self._recorder.start_recording(video_file_name)

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

    def _start_state_is_valid(self, maze_size, start_loc, target_loc):
        """
        This function ensures that the locations are in the maze
        :param maze_size: tuple of the maze size (x,y)
        :param start_state: dictionary - {start_loc : tuple(3), target_loc : tuple(3)}
        """
        # s_loc = start_state["start_loc"]
        # t_loc = start_state["target_loc"]
        # # TODO - fix to make sure coordinates are >1. also make MazeEnv member function
        # if s_loc[0] > maze_size[0] or s_loc[1] > maze_size[1] \
        #         or t_loc[0] > maze_size[0] or t_loc[1] > maze_size[1]:
        #     return False

        return True



