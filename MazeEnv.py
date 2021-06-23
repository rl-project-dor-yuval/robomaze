from typing import Optional, Tuple
import gym
from gym import error, spaces, utils
from gym.spaces import Box, Dict
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

    recording_video_size: Tuple[int, int] = (400, 400)
    video_skip_frames: int = 4
    zoom: float = 1.2  # is also relative to maze size

    # private members:
    _collision_manager: CollisionManager
    _maze: Maze
    _ant: Ant
    _recorder: Recorder

    _start_loc: Tuple[float, float, float]
    _target_loc: Tuple[float, float, float]

    _physics_server: int
    _connectionUid: int

    def __init__(self,
                 maze_size=MazeSize.MEDIUM,
                 maze_map: np.ndarray = None,
                 tile_size=0.1,
                 start_loc=(1, 1),
                 target_loc=(3, 3),
                 rewards: Rewards = Rewards(),
                 timeout_steps: int = 0,
                 show_gui: bool = False,
                 observations: ObservationsDefinition = ObservationsDefinition(), ):
        """
        :param maze_size: the size of the maze from : {MazeSize.SMALL, MazeSize.MEDIUM, MazeSize.LARGE}
        :param maze_map: a boolean numpy array of the maze. shape must be maze_size ./ tile_size.
         if no value is passed, an empty maze is made, means there are only edges.
        :param tile_size: the size of each point in the maze passed by maze map. determines the
         resolution of the maze
        :param start_loc: the location the ant starts at
        :param target_loc: the location of the target sphere center
        :param rewards: definition of reward values for events
        :param timeout_steps: maximum steps until getting timeout reward
         (if a timeout reward is defined)
        :param show_gui: if set to true, the simulation will be shown in a GUI window
        :param observations: definition of the desired observations for the agent

        :return: Maze Environment object

        Initializing environment object
        """
        if maze_map is not None and np.any(maze_size != np.dot(maze_map.shape, tile_size)):
            raise Exception("maze_map and maze_size mismatch. maze_map size is {map_size}, "
                            "maze size is {maze_size}, tile_size is {tile_size}.\n "
                            "note that map_size must be  maze_size / tile_size.".format(map_size=maze_map.shape,
                                                                                        maze_size=maze_size,
                                                                                        tile_size=tile_size))
        if not self._start_state_is_valid(maze_size, start_loc, target_loc):
            raise Exception("Start state is invalid, ant or target in a wrong position")
        if timeout_steps < 0:
            raise Exception("timeout_steps value must be positive or zero for no limitation")

        self.is_reset = False
        self.step_count = 0
        self.episode_count = 0
        self._start_loc = (start_loc[0], start_loc[1], _ANT_START_Z_COORD)
        self._target_loc = (target_loc[0], target_loc[1], 0)
        self.rewards = rewards
        self.timeout_steps = timeout_steps

        self.action_space = Box(low=-1, high=1, shape=(8,))

        observations_bounds_low, observations_bounds_high = self._get_observation_bounds(maze_size)
        self.observation_space = Box(observations_bounds_low, observations_bounds_high)

        # setup simulation:
        if show_gui:
            self._physics_server = p.GUI
        else:
            self._physics_server = p.DIRECT

        self._connectionUid = p.connect(self._physics_server)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # load maze:
        self._maze = Maze(maze_size, maze_map, tile_size, self._target_loc)

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
        if not self.action_space.contains(action):
            raise Exception("Expected shape (8,) and value in [-1,1] ")

        # initialize return values:
        observation = self._get_observation()
        reward = 0
        is_done = False
        info = None  # TODO: handle?

        # pass actions through the ant object:
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

    def _get_observation(self):
        """in the future the observation space is going to be configurable,
            right now its just a 21D vector. see self._get_observation_bounds
            for detail"""
        observation = np.zeros(self.observation_space.shape)

        observation[np.array([0, 1, 2, 3, 20])] = self._ant.get_pos_vel_and_facing_direction()
        observation[4:20] = self._ant.get_joint_state()

        return observation

    @staticmethod
    def _start_state_is_valid(maze_size, start_loc, target_loc):
        """
        This function ensures that the locations are in the maze
        :param maze_size: tuple of the maze size (x,y)
        :param start_state: dictionary - {start_loc : tuple(3), target_loc : tuple(3)}
        """
        # s_loc = start_state["start_loc"]
        # t_loc = start_state["target_loc"]
        # # TODO - fix to make sure coordinates are >1. also make MazeEnv member function
        # # TODO make sure that did not start on a maze tile
        # if s_loc[0] > maze_size[0] or s_loc[1] > maze_size[1] \
        #         or t_loc[0] > maze_size[0] or t_loc[1] > maze_size[1]:
        #     return False

        return True

    @staticmethod
    def _get_observation_bounds(maze_size):
        # ant position 2d:
        observations_bounds_high = [maze_size[0] / 2, maze_size[1] / 2]

        # ant velocity 2d:
        observations_bounds_high.append(np.finfo(np.float32).max)
        observations_bounds_high.append(np.finfo(np.float32).max)

        # ant facing direction 1d
        observations_bounds_high.append(math.pi)

        # joint position 8d + joint velocity 8d:
        observations_bounds_high += [1] * 16

        observations_bounds_high = np.array(observations_bounds_high)
        observations_bounds_low = -observations_bounds_high

        return observations_bounds_low, observations_bounds_high
