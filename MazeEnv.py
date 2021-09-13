from typing import Optional, Tuple
import gym
from gym.spaces import Box, Dict
from gym.utils import seeding
import pybullet
from pybullet_utils import bullet_client as bc
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
    rewards: Rewards

    timeout_steps: int
    step_count: int
    is_reset: bool
    episode_count: int

    recording_video_size: Tuple[int, int] = (400, 400)
    video_skip_frames: int = 4
    zoom: float = 1.2  # is also relative to maze size

    _collision_manager: CollisionManager
    _maze: Maze
    _ant: Ant
    _recorder: Recorder

    _start_loc: Tuple[float, float, float]
    _target_loc: Tuple[float, float, float]

    _physics_server: int
    _pclient: bc.BulletClient

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
        # will raise an exception if something is wrong:
        self._check_start_state(maze_size, start_loc, target_loc)

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
        # self.observation_space = Box(observations_bounds_low, observations_bounds_high)
        self.observation_space = Box(-np.inf, np.inf, (21,))

        # setup simulation:
        if show_gui:
            self._physics_server = pybullet.GUI
        else:
            self._physics_server = pybullet.DIRECT

        self._pclient = bc.BulletClient(connection_mode=self._physics_server)
        self._pclient.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._pclient.setGravity(0, 0, -10)
        self._pclient.configureDebugVisualizer(self._pclient.COV_ENABLE_GUI, False)  # dont show debugging windows

        # load maze:
        self._maze = Maze(self._pclient, maze_size, maze_map, tile_size, self._target_loc)

        # load ant robot:
        self._ant = Ant(self._pclient, self._start_loc)

        # create collision detector and pass relevant uids:
        maze_uids, target_sphere_uid = self._maze.get_maze_objects_uids()
        self._collision_manager = CollisionManager(self._pclient,
                                                   maze_uids,
                                                   target_sphere_uid,
                                                   self._ant.uid)

        # setup camera for a bird view:
        self._pclient.resetDebugVisualizerCamera(cameraDistance=self._maze.maze_size[1] / self.zoom,
                                     cameraYaw=0,
                                     cameraPitch=-89.9,
                                     cameraTargetPosition=[self._maze.maze_size[0] / 2, self._maze.maze_size[1] / 2, 0])

        self._recorder = Recorder(self._pclient,
                                  maze_size=maze_size,
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
        info = dict()  # TODO: handle?

        # pass actions through the ant object:
        self._ant.action(action)

        # run simulation step
        self._pclient.stepSimulation()

        self.step_count += 1

        # reward and is_done update:
        # check for ant collision in the last step and update reward:
        hit_target, hit_maze = self._collision_manager.check_ant_collisions()
        if hit_target:
            reward += self.rewards.target_arrival
            is_done = True
        if hit_maze:
            reward += self.rewards.collision
            is_done = True
        # check for timeout:
        if self.timeout_steps != 0 and self.step_count >= self.timeout_steps:
            reward += self.rewards.timeout
            is_done = True

        # handle recording
        if self._recorder.is_recording and \
                self.step_count % self.video_skip_frames == 0:
            self._recorder.insert_current_frame()

        # if done and recording save video
        if is_done and self._recorder.is_recording:
            self._recorder.save_recording_and_reset()

        return observation, reward, is_done, info

    def reset(self, create_video=False, video_path=None, reset_episode_count=False):
        """
        reset the environment for the next episode
        :param create_video: weather to create video file from the next episode
        :param video_path: path to the video file. if None then the file will be saved
                            to the default path at "/videos/date-time/episode#.mp4
        :param reset_episode_count: weather to reset the MazeEnv.episode_count value

        :return observation for the zeroth time step
        """
        # move ant to start position:
        self._ant.reset()

        # handle recording (save last episode if needed)
        if self._recorder.is_recording:
            self._recorder.save_recording_and_reset()
        if create_video:
            if video_path is None:
                video_file_name = "episode" + str(self.episode_count) + ".avi"
                self._recorder.start_recording(video_file_name)
            else:
                self._recorder.start_recording(video_path, custom_path=True)

        # update self state:
        self.episode_count = 0 if reset_episode_count else self.episode_count
        self.episode_count += 1
        self.step_count = 0
        self.is_reset = True

        return self._get_observation()

    def _get_observation(self):
        """in the future the observation space is going to be configurable,
            right now its just a 21D vector. see self._get_observation_bounds
            for detail"""
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)

        observation[np.array([0, 1, 2, 3, 20])] = self._ant.get_pos_vel_and_facing_direction()
        observation[4:20] = self._ant.get_joint_state()

        return observation

    @staticmethod
    def _check_start_state(maze_size, start_loc, target_loc):
        """
        This function ensures that the locations are inside the maze.
        It does not handle the cases where the ant or target are on
        a maze tile or the maze is unsolvable.
        """
        min_x = min_y = 1
        max_x, max_y = (maze_size[0] - 1), (maze_size[1] - 1)
        target_loc = tuple(target_loc)
        start_loc = tuple(start_loc)

        if start_loc < (min_x, min_y) or start_loc > (max_x, max_y) or \
                target_loc < (min_x, min_y) or target_loc > (max_x, max_y):
            raise Exception(f"Start location and target location must be at least "
                            f"1 unit away from maze boundries which is {min_x} < x < {max_x} "
                            f"and {min_y} < y < {max_y} for this maze size")

    @staticmethod
    def _get_observation_bounds(maze_size):
        # ant position 2d:
        observations_bounds_high = [maze_size[0] / 2, maze_size[1] / 2]

        # ant velocity 2d:
        observations_bounds_high.append(np.inf)
        observations_bounds_high.append(np.inf)

        # ant facing direction 1d
        observations_bounds_high.append(math.pi)

        # joint position 8d
        observations_bounds_high += [1] * 8

        # joint velocity 8d:
        observations_bounds_high += [np.inf] * 8

        observations_bounds_high = np.array(observations_bounds_high, dtype=np.float32)
        observations_bounds_low = -observations_bounds_high

        return observations_bounds_low, observations_bounds_high
