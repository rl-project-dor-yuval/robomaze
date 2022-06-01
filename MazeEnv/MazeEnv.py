import time
from typing import Optional, Tuple, List
import gym
from gym.spaces import Box, Dict
import pybullet
from pybullet_utils import bullet_client as bc
import pybullet_data
import numpy as np
import math
import os
from MazeEnv.Recorder import Recorder
from MazeEnv.EnvAttributes import Rewards, MazeSize
from MazeEnv.CollisionManager import CollisionManager
from MazeEnv.Ant import Ant
from MazeEnv.Maze import Maze

_ANT_START_Z_COORD = 0.7  # the height the ant starts at


class MazeEnv(gym.Env):
    rewards: Rewards

    timeout_steps: int
    step_count: int
    is_reset: bool
    episode_count: int
    xy_in_obs: bool
    rewards: Rewards
    done_on_collision: bool

    recording_video_size: Tuple[int, int] = (200, 200)
    video_skip_frames: int = 1
    zoom: float = 1.1  # is also relative to maze size

    _collision_manager: CollisionManager
    _maze: Maze
    _ant: Ant
    _subgoal_marker: int
    _recorder: Recorder

    _start_loc: List
    _target_loc: List
    hit_target_epsilon: float
    max_goal_velocity: float

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
                 xy_in_obs: bool = True,
                 hit_target_epsilon=0.4,
                 done_on_collision=True,
                 success_steps_before_done: int = 1,
                 noisy_ant_initialization=False,
                 goal_max_velocity: float = np.inf,
                 optimize_maze_boarders: bool = True,
                 sticky_actions=5):
        """
        :param maze_size: the size of the maze from : {MazeSize.SMALL, MazeSize.MEDIUM, MazeSize.LARGE}
        :param maze_map: a boolean numpy array of the maze. shape must be maze_size ./ tile_size.
         if no value is passed, an empty maze is made, means there are only edges.
        :param tile_size: the size of each point in the maze passed by maze map. determines the
         resolution of the maze
        :param start_loc: the location the ant starts at
        :param target_loc: the location of the target sphere center
        :param rewards: definition of reward values for events
        :param timeout_steps: maximum steps until getting timeout reward and episode ends
         (if a timeout reward is defined)
        :param show_gui: if set to true, the simulation will be shown in a GUI window
        :param xy_in_obs: Weather to return the X and Y location of the robot in the observation.
                if True, the two first elements of the observation are X and Y
        :param done_on_collision: if True, episodes ends when the ant collides with the wall
        :param success_steps_before_done: number of steps the ant has to be in the target location
          (within epsilon distance) to end episode
        :type noisy_ant_initialization: if True, the ant will start with a random joint state and with
         a noisy orientation at each reset
        :param goal_max_velocity: optional velocity limit to consider reaching goal
        :param optimize_boarders: if True, collision detection is checked only on boarder of
         free areas on the map
        :param sticky_actions: for how many simulation steps to repeat an action.


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

        self.maze_size = maze_size
        self._start_loc = [start_loc[0], start_loc[1], _ANT_START_Z_COORD]
        self._target_loc = [target_loc[0], target_loc[1], 0]
        self.rewards = rewards
        self.timeout_steps = timeout_steps
        self.xy_in_obs = xy_in_obs
        self.hit_target_epsilon = hit_target_epsilon
        self.done_on_collision = done_on_collision
        self.success_steps_before_done = success_steps_before_done
        self.max_goal_velocity = goal_max_velocity
        self.noisy_ant_initialization = noisy_ant_initialization
        self.sticky_actions = sticky_actions

        self.is_reset = False
        self.step_count = 0
        self.episode_count = 0
        self.success_steps = 0

        self.action_space = Box(low=-1, high=1, shape=(8,))

        obs_space_size = 30 if xy_in_obs else 28
        self.observation_space = Box(-np.inf, np.inf, (obs_space_size,))

        # setup simulation:
        if show_gui:
            self._physics_server = pybullet.GUI
        else:
            self._physics_server = pybullet.DIRECT

        curr_dirname = os.path.dirname(__file__)
        data_dir = os.path.join(curr_dirname, 'data')
        self._pclient = bc.BulletClient(connection_mode=self._physics_server)
        self._pclient.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._pclient.setAdditionalSearchPath(curr_dirname)
        self._pclient.setAdditionalSearchPath(data_dir)
        self._pclient.setGravity(0, 0, -10)
        self._pclient.configureDebugVisualizer(self._pclient.COV_ENABLE_GUI, False)  # dont show debugging windows

        # load maze:
        self._maze = Maze(self._pclient, maze_size, maze_map, tile_size, self._target_loc, optimize_maze_boarders)

        # load ant robot:
        self._ant = Ant(self._pclient, self._start_loc)

        # create collision detector and pass relevant uids:
        maze_uids, target_sphere_uid, floorUid = self._maze.get_maze_objects_uids()
        self._collision_manager = CollisionManager(self._pclient,
                                                   maze_uids,
                                                   target_sphere_uid,
                                                   self._ant.uid,
                                                   floorUid)

        # create subgoal marker:
        self._subgoal_marker = self._pclient.loadURDF("goalSphere.urdf",
                                                      basePosition=(0, 0, 0),
                                                      globalScaling=0.5)
        self._pclient.changeVisualShape(self._subgoal_marker, -1, rgbaColor=[0, 0, 0, 0])
        self._pclient.setCollisionFilterGroupMask(self._subgoal_marker, -1, 0, 0)  # disable collisions

        # setup camera for a bird view:
        self._pclient.resetDebugVisualizerCamera(cameraDistance=self._maze.maze_size[1] / self.zoom,
                                                 cameraYaw=0,
                                                 cameraPitch=-89.9,
                                                 cameraTargetPosition=[self._maze.maze_size[0] / 2,
                                                                       self._maze.maze_size[1] / 2, 0])

        self._recorder = Recorder(self._pclient,
                                  maze_size=maze_size,
                                  video_size=self.recording_video_size,
                                  zoom=self.zoom,
                                  fps=40)

    def step(self, action):
        if not self.is_reset:
            raise Exception("MazeEnv.reset() must be called before before MazeEnv.step()")
        # if not self.action_space.contains(action):
        #     raise Exception("Expected shape (8,) and value in [-1,1] ")

        # perform step: pass actions through the ant object and run simulation step:
        for _ in range(self.sticky_actions):  # loop incase we want sticky actions
            self._ant.action(action)
            self._pclient.stepSimulation()

        self.step_count += 1

        # resolve observation:
        observation = self._get_observation()
        ant_xy = observation[0:2]
        if not self.xy_in_obs:
            observation = observation[2:]

        # check status and resolve reward and is_done:
        reward = 0
        is_done = False
        info = dict(success=False, fell=False, hit_maze=False, timeout=False)

        if self._collision_manager.check_hit_floor():
            is_done = info['fell'] = True
            reward += self.rewards.fall
        if self._collision_manager.check_hit_maze():
            is_done = self.done_on_collision
            info['hit_maze'] = True
            reward += self.rewards.collision

        target_loc_xy = np.array([self._target_loc[0], self._target_loc[1]])
        target_distance = np.linalg.norm(target_loc_xy - ant_xy)

        reward += self.rewards.compute_target_distance_reward(target_distance=target_distance)

        if target_distance < self.hit_target_epsilon:
            # check if meets velocity condition:
            vx, vy = observation[3], observation[4]
            if np.sqrt(vx ** 2 + vy ** 2) < self.max_goal_velocity:
                info['success'] = True
                reward += self.rewards.target_arrival
                self.success_steps += 1
                if self.success_steps >= self.success_steps_before_done:
                    is_done = True

        if self.timeout_steps != 0 and self.step_count >= self.timeout_steps:
            is_done = info['timeout'] = True
            reward += self.rewards.timeout

        reward += self.rewards.idle

        # handle recording
        if self._recorder.is_recording and \
                self.step_count % self.video_skip_frames == 0:
            self._recorder.insert_current_frame()

        # if done and recording save video
        if is_done and self._recorder.is_recording:
            self._recorder.save_recording_and_reset()

        # print("-----------------")
        # print(f"action {action}")
        # print(f"reward {reward}")
        # print(f"is_done {is_done}")

        return observation, reward, is_done, info

    def reset(self, create_video=False, video_path=None, reset_episode_count=False):
        """
        :param create_video: weather to create video file from the next episode
        :param video_path: path to the video file. if None then the file will be saved
                            to the default path at "/videos/date-time/episode#.mp4
        :param reset_episode_count: weather to reset the MazeEnv.episode_count value

        :return: observation for the the first time step

        reset the environment for the next episode
        """
        # move ant to start position:
        self._ant.reset(self.noisy_ant_initialization)

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
        self.success_steps = 0
        self.is_reset = True

        observation = self._get_observation()
        if not self.xy_in_obs:
            observation = observation[2:]
        return observation

    def set_subgoal_marker(self, position=(0, 0), visible=True):
        """
        put a marker on the given position
        :param visible: set to false in order to remove the marker
        :param position: the position of the marker
        """
        if visible:
            position = (*position, 0)
            self._pclient.changeVisualShape(self._subgoal_marker, -1, rgbaColor=[0.5, 0.5, 0.5, 1])
            self._pclient.resetBasePositionAndOrientation(self._subgoal_marker, position, [0, 0, 0, 1])
        else:
            self._pclient.changeVisualShape(self._subgoal_marker, -1, rgbaColor=[0, 0, 0, 0])

    def set_target_loc(self, new_loc):
        """
        set the target location. Call this Only Before reset()!
        :param position: the position of the target
        """
        self._target_loc[0], self._target_loc[1] = new_loc

        _, target_uid, _ = self._maze.get_maze_objects_uids()
        _, old_orientation = self._pclient.getBasePositionAndOrientation(target_uid)
        self._pclient.resetBasePositionAndOrientation(target_uid, self._target_loc, old_orientation)

    def set_start_loc(self, start_loc):
        """
        change the start location of the ant in the next reset
        :param start_loc: tuple of the start location
        :return: None
        """
        self._check_start_state(self._maze.maze_size, start_loc, self._target_loc)
        self._ant.start_position[0], self._ant.start_position[1] = start_loc[0], start_loc[1]
        self._start_loc[0], self._start_loc[1] = start_loc[0], start_loc[1]

    def set_timeout_steps(self, timeout_steps):
        """
        change the amount of steps for timeout. The new value applies from the next step
        :param timeout_steps: timeout steps for next step
        :return: None
        """
        self.timeout_steps = timeout_steps

    def _get_observation(self):
        """
        get 30D/28D observation vector according to use of x,y
        observation consists of:
        [ 0:2 - ant COM position (x,y,z),
          3:5 - ant COM velocity (x,y,z),
          6:8 - ant euler orientation [Roll, Pitch, Yaw],
          9:11 - ant angular velocity (x,y,z),
          12:19 - ant joint position (8 joints),
          20:27 - ant joint velocities (8 joints),
          28 - relative distance from target,
          29 - relative angle to target (in radians) ]
        """
        # if xy not in observation it will be cut later
        observation = np.zeros(30, dtype=np.float32)

        observation[0:12] = self._ant.get_pos_orientation_velocity()
        observation[12:28] = self._ant.get_joint_state()

        # last two elements are angel and distance from target
        ant_loc = observation[0:2]
        target_loc = np.array(self._target_loc[0:2])
        relative_target = target_loc - ant_loc

        observation[28] = np.linalg.norm(relative_target)
        observation[29] = np.arctan2(relative_target[1], relative_target[0])

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

    def set_position_control(self, position_control: bool = True):
        """set the ant to position control if true, or torque control if false"""
        if position_control:
            print("WARNING: setting ant to position control! make sure you know what you are doing! "
                  "and please don't mix position and torque control in the same episode.")
        self._ant.set_position_control(position_control)
