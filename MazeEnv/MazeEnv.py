import time
from typing import Optional, Tuple, List, Union
import gym
from gym.spaces import Box, Dict
import pybullet
from pybullet_utils import bullet_client as bc
import pybullet_data
import numpy as np
import math
import os
from MazeEnv.Recorder import Recorder
from MazeEnv.EnvAttributes import Rewards, MazeSize, Workspace
from MazeEnv.CollisionManager import CollisionManager
from MazeEnv.Ant import Ant, RobotBase
from MazeEnv.Rex import Rex
from MazeEnv.Bipedal import Bipedal
from MazeEnv.Maze import Maze


robot_types = {"Ant": Ant, "Rex": Rex, "Bipedal": Bipedal}


class MazeEnv(gym.Env):
    rewards: Rewards

    timeout_steps: int
    step_count: int
    is_reset: bool
    episode_count: int
    xy_in_obs: bool
    rewards: Rewards
    done_on_collision: bool

    recording_video_size: Tuple[int, int] = (250, 250)
    video_skip_frames: int = 1
    zoom: float = 1.05  # is also relative to maze size

    _collision_manager: CollisionManager
    _maze: Maze
    _robot: RobotBase
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
                 workspace: Workspace = Workspace(),
                 rewards: Rewards = Rewards(),
                 timeout_steps: int = 0,
                 show_gui: bool = False,
                 xy_in_obs: bool = True,
                 hit_target_epsilon=0.4,
                 target_heading_epsilon=np.pi,
                 done_on_collision=True,
                 done_on_goal_reached=True,
                 success_steps_before_done: int = 1,
                 noisy_robot_initialization=False,
                 goal_max_velocity: float = np.inf,
                 optimize_maze_boarders: bool = True,
                 sticky_actions=8,
                 robot_type: str = 'Ant',):
        """
        # TODO: Update docstring
        :param maze_size: the size of the maze from : {MazeSize.SMALL, MazeSize.MEDIUM, MazeSize.LARGE}
        :param maze_map: a boolean numpy array of the maze. shape must be maze_size ./ tile_size.
         if no value is passed, an empty maze is made, means there are only edges.
        :param tile_size: the size of each point in the maze passed by maze map. determines the
         resolution of the maze
        :param start_loc: the location the robot starts at
        :param target_loc: the location of the target sphere center
        :param rewards: definition of reward values for events
        :param timeout_steps: maximum steps until getting timeout reward and episode ends
         (if a timeout reward is defined)
        :param show_gui: if set to true, the simulation will be shown in a GUI window
        :param xy_in_obs: Weather to return the X and Y location of the robot in the observation.
                if True, the two first elements of the observation are X and Y
        :param done_on_collision: if True, episodes ends when the robot collides with the wall
        :param done_on_goal_reached
        :param success_steps_before_done: number of steps the robot has to be in the target location
          (within epsilon distance) to end episode in case done_on_goal_reached is True (otherwise ignored)
        :type noisy_robot_initialization: if True, the robot will start with a random joint state and with
         a noisy orientation at each reset
        :param goal_max_velocity: optional velocity limit to consider reaching goal
        :param optimize_boarders: if True, collision detection is checked only on boarder of
         free areas on the map
        :param sticky_actions: for how many simulation steps to repeat an action.
        :param robot_type: string determines the the robot type - 'Ant' or 'Rex'.
        Initializing environment object
        """

        if maze_map is not None and np.any(maze_size != np.dot(maze_map.shape, tile_size)):
            raise Exception("maze_map and maze_size mismatch. maze_map size is {map_size}, "
                            "maze size is {maze_size}, tile_size is {tile_size}.\n "
                            "note that map_size must be  maze_size / tile_size.".format(map_size=maze_map.shape,
                                                                                        maze_size=maze_size,
                                                                                        tile_size=tile_size))
        # will raise an exception if something is wrong:
        self._check_start_state(maze_size, workspace.start_loc_tuple(), workspace.goal_loc_tuple())

        if timeout_steps < 0:
            raise Exception("timeout_steps value must be positive or zero for no limitation")

        self.maze_size = maze_size
        self.workspace = workspace
        self.rewards = rewards
        self.timeout_steps = timeout_steps
        self.xy_in_obs = xy_in_obs
        self.hit_target_epsilon = hit_target_epsilon
        self.target_heading_epsilon = target_heading_epsilon
        self.done_on_collision = done_on_collision
        self.done_on_goal_reached = done_on_goal_reached
        self.success_steps_before_done = success_steps_before_done
        self.max_goal_velocity = goal_max_velocity
        self.noisy_robot_initialization = noisy_robot_initialization
        self.sticky_actions = sticky_actions

        self.is_reset = False
        self.step_count = 0
        self.episode_count = 0
        self.success_steps = 0

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
        target_loc3d = workspace.goal_loc_tuple() + (0,)
        self._maze = Maze(self._pclient, maze_size, maze_map, tile_size, target_loc3d, workspace.goal_heading,
                          optimize_maze_boarders)

        # load robot:
        robot_loc_2d = workspace.start_loc_tuple()
        if not robot_type in robot_types.keys():
            raise Exception("robot_type must be one of {}, but was {}".format(robot_types.keys(), robot_type))
        self._robot = robot_types[robot_type](self._pclient, robot_loc_2d, workspace.start_heading,)

        # set Action & State space
        self.action_space = Box(low=-1, high=1, shape=(self._robot.get_action_dim(),))
        obs_space_size = self._robot.get_state_dim()
        if not self.xy_in_obs:
            obs_space_size -= 2
        self.observation_space = Box(-np.inf, np.inf, (obs_space_size,))

        # create collision detector and pass relevant uids:
        maze_uids, target_sphere_uid, floorUid = self._maze.get_maze_objects_uids()
        self._collision_manager = CollisionManager(self._pclient,
                                                   maze_uids,
                                                   target_sphere_uid,
                                                   self._robot.uid,
                                                   floorUid)

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
                                  fps=60)

    def step(self, action):
        if not self.is_reset:
            raise Exception("MazeEnv.reset() must be called before before MazeEnv.step()")
        # if not self.action_space.contains(action):
        #     raise Exception("Expected shape (8,) and value in [-1,1] ")

        # perform step: pass actions through the robot object and run simulation step:
        for _ in range(self.sticky_actions):  # loop incase we want sticky actions
            self._robot.action(action)
            self._pclient.stepSimulation()

        self._robot.update_direction_pointer(visible=True)
        # I left the option to choose not to show robot direction pointer in case we want this. just pass false.

        self.step_count += 1

        # resolve observation:
        observation = self._get_observation()
        robot_xy = observation[0:2]
        robot_z = observation[2]
        vx, vy = observation[3], observation[4]
        robot_heading_diff = observation[14]

        observation = self.clean_xy_from_obs_if_needed(observation)

        # check status and resolve reward and is_done:
        reward = 0
        is_done = False
        info = {'success': False, 'fell': False, 'hit_maze': False, 'TimeLimit.truncated': False}

        if self._collision_manager.check_hit_floor():
            is_done = info['fell'] = True
            reward += self.rewards.fall
        if self._collision_manager.check_hit_maze():
            is_done = self.done_on_collision
            info['hit_maze'] = True
            reward += self.rewards.collision

        goal_loc_xy = np.array(self.workspace.goal_loc_tuple())
        goal_distance = np.linalg.norm(goal_loc_xy - robot_xy)

        reward += self.rewards.compute_target_distance_reward(target_distance=goal_distance)
        reward += self.rewards.compute_rotation_reward(rotation_diff=robot_heading_diff)
        reward += robot_z * self.rewards.height_weight

        # check if goal is reached and update info/reward/is_done:
        if goal_distance < self.hit_target_epsilon and abs(robot_heading_diff) < self.target_heading_epsilon:
            # check if meets velocity condition:
            if np.sqrt(vx ** 2 + vy ** 2) < self.max_goal_velocity:
                info['success'] = True
                reward += self.rewards.target_arrival
                self.success_steps += 1
                if self.success_steps >= self.success_steps_before_done and \
                        self.done_on_goal_reached:
                    is_done = True

        if self.timeout_steps != 0 and self.step_count >= self.timeout_steps:
            if info['success']:
                is_done = True
            else:
                is_done = info['TimeLimit.truncated'] = True
                reward += self.rewards.timeout

        reward += self.rewards.idle

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
        :param create_video: weather to create video file from the next episode
        :param video_path: path to the video file. if None then the file will be saved
                            to the default path at "/videos/date-time/episode#.mp4
        :param reset_episode_count: weather to reset the MazeEnv.episode_count value

        :return: observation for the the first time step

        reset the environment for the next episode
        """
        # move robot to start position:
        self._robot.reset(self.noisy_robot_initialization)

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
        observation = self.clean_xy_from_obs_if_needed(observation)
        return observation

    def set_subgoal_marker(self, position=(0, 0), heading=0, visible=True):
        """
        put a marker on the given position
        :param position: the position of the marker
        :param heading: heading of the pointer above the marker
        :param visible: set to false in order to remove the marker
        """
        self._maze.set_subgoal_marker(position, heading, visible)

    def set_workspace(self, workspace: Workspace):
        """
        sets new workspace. please use it just before calling reset.
        It shouldn't have any effect before reset
        """
        self._check_start_state(self.maze_size, workspace.start_loc_tuple(), workspace.goal_loc_tuple())
        self.workspace = workspace

        goal_loc_3d = workspace.goal_loc_tuple() + (0,)
        self._maze.set_new_goal(goal_loc_3d, workspace.goal_heading)

        robot_loc_2d = workspace.start_loc_tuple()
        self._robot.set_start_state(robot_loc_2d, workspace.start_heading)

    # TODO: Remove those methods after checking that everything is working fine:
    # def set_target_loc_and_heading(self, new_loc, new_heading=None):
    #     """
    #     set the target location. Call this Only Before reset()!
    #     :param new_loc: the position of the target
    #     :param new_heading: the heading at the target. optional if not given the heading will be 0
    #     """
    #     self._target_loc[0], self._target_loc[1] = new_loc
    #     if new_heading is not None:
    #         self._target_heading = new_heading
    #     else:
    #         self._target_heading = 0
    #
    #     # physically move and rotate goal
    #     self._maze.set_new_goal(self._target_loc, self._target_heading)
    #
    # def set_start_loc(self, start_loc):
    #     """
    #     change the start location of the robot in the next reset
    #     :param start_loc: tuple of the start location
    #     :return: None
    #     """
    #     self._check_start_state(self._maze.maze_size, start_loc, self._target_loc)
    #     self._robot.start_position[0], self._robot.start_position[1] = start_loc[0], start_loc[1]
    #     self._start_loc[0], self._start_loc[1] = start_loc[0], start_loc[1]
    #
    # def get_target_loc(self):
    #     """
    #     :return: the target location
    #     """
    #     return self._target_loc[:2]

    def set_timeout_steps(self, timeout_steps):
        """
        change the amount of steps for timeout. The new value applies from the next step
        :param timeout_steps: timeout steps for next step
        :return: None
        """
        self.timeout_steps = timeout_steps

    def _get_observation(self):
        """
        get 31D/29D observation vector according to use of x,y
        observation consists of:
        [ 0:2 -   robot COM position (x,y,z),
          3:5 -   robot COM velocity (x,y,z),
          6:8 -   robot euler orientation [Roll, Pitch, Yaw],
          9:11 -  robot angular velocity (x,y,z),
          12 - relative distance from target, # TODO can be computed in stepper instead of here
          13 - relative angle to target (in radians), # TODO can be computed in stepper instead of here
          14 - angle between rotation of the robot and target heading # TODO we probably don't need this anymores
          n robot joint states
          n robot joint velocities]
        """
        # if xy not in observation it will be cut later
        obs_dim = self._robot.get_state_dim()
        observation = np.zeros(obs_dim, dtype=np.float32)

        # first 12 elements
        observation[0:12] = self._robot.get_pos_orientation_velocity()

        # next two elements are angel and distance from target
        robot_loc = observation[0:2]
        target_loc = np.array(self.workspace.goal_loc_tuple())
        relative_target = target_loc - robot_loc
        observation[12] = np.linalg.norm(relative_target)
        observation[13] = np.arctan2(relative_target[1], relative_target[0])

        # last element is rotation angle between robot and target heading
        robot_rotation = observation[8]
        rotation_diff = self.workspace.goal_heading - robot_rotation
        observation[14] = self.compute_signed_rotation_diff(rotation_diff)

        # setting observation according to robot dim
        observation[15:] = self._robot.get_joint_state()

        return observation

    def get_joint_state_dim(self):
        return self._robot.get_joint_state_dim()

    def clean_xy_from_obs_if_needed(self, obs):
        if not self.xy_in_obs:
            return self.clean_xy_from_obs(obs)
        return obs

    @staticmethod
    def clean_xy_from_obs(obs):
        """
        remove xy and heading from observation
        :param obs: the observation
        :return: the observation without xy and heading
        """
        # indices of xy are 0, 1 and index of heading is 8
        return obs[2:]

    @staticmethod
    def _check_start_state(maze_size, start_loc, target_loc):
        """
        This function ensures that the locations are inside the maze.
        It does not handle the cases where the robot or target are on
        a maze tile or the maze is unsolvable.
        """
        min_x = min_y = 0.7
        max_x, max_y = (maze_size[0] - 0.7), (maze_size[1] - 0.7)
        target_loc = tuple(target_loc)
        start_loc = tuple(start_loc)

        if start_loc < (min_x, min_y) or start_loc > (max_x, max_y) or \
                target_loc < (min_x, min_y) or target_loc > (max_x, max_y):
            raise Exception(f"Start location and target location must be at least "
                            f"0.7 units away from maze boundries which is {min_x} < x < {max_x} "
                            f"and {min_y} < y < {max_y} for this maze size")

    @staticmethod
    def compute_signed_rotation_diff(rotation_diff):
        """
        given a raw angles difference, cast it to a signed angle between -pi and pi.
        original diff may be more than pi or less than -pi.
        """
        unsigned_diff = np.abs(rotation_diff) % 360
        if unsigned_diff > 180:
            unsigned_diff = 360 - unsigned_diff

        rotation_sign = 1 if (0 <= rotation_diff <= 180) or (-180 >= rotation_diff >= -360) else -1

        return rotation_sign * unsigned_diff
