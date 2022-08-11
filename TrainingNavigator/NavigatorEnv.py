from typing import Union

import gym
from gym.spaces import Box
import MazeEnv.MazeEnv as mz
from Training.TransformObservation import transform_to_stepper_obs
from TrainingNavigator.StepperAgent import StepperAgent
import numpy as np
import math
import time
from MazeEnv.EnvAttributes import Rewards


def cart2pol(vec):
    r = np.linalg.norm(vec)
    theta = np.arctan2(vec[1], vec[0])
    return np.array([r, theta])


def pol2cart(vec):
    x = vec[0] * np.cos(vec[1])
    y = vec[0] * np.sin(vec[1])
    return np.array([x, y])


class NavigatorEnv(gym.Env):
    """
    NavigatorEnv is a wrapper around MazeEnv and a stepper agent so that an s here is a command to the stepper.
    Generally the navigator's purpose is given a certain Goal, generate a sequence of subgoals (Navigator actions)
    all the way to it, whereas the stepperAgent knows how to reach close subgoals.
    """

    def __init__(self,
                 maze_env: mz.MazeEnv = None,
                 maze_env_kwargs: dict = None,
                 stepper_agent: Union[StepperAgent, str] = None,
                 max_stepper_steps=200,
                 max_steps=50,
                 stepper_radius_range=(0.6, 2.5),
                 epsilon_to_hit_subgoal=0.25,
                 max_vel_in_subgoal=9999,
                 rewards: Rewards = Rewards(),
                 done_on_collision=False,
                 normalize_observations=True,
                 wall_hit_limit=-1):
        """
        :param maze_env: mz.MazeEnv object - with observation space of 30d - with robot x,y location
        :param maze_env_kwargs: if maze_env is None, then this is used to create a new maze_env,
                                otherwise this is ignored
        :param stepper_agent: stepper agent object
        :param max_stepper_steps: maximum steps allowed to the stepper towards subgoal.
        :param max_steps: max steps for navigator episode
        :param stepper_radius_range: defined by the reachable radius of the stepper
        :param epsilon_to_hit_subgoal: minimal distance to subgoal to be considered as goal arrival
        :param max_vel_in_subgoal: maximal vertical velocity in subgoal to be considered as goal arrival
        :param rewards: Rewards object, defines the reward for each event
        :param done_on_collision: weather to kill the robot when colliding the wall
        :param normalize_observations
        :param wall_hit_limit: if wall_hit_limit is > 0, and done in collision is True, then the episode is done if
                     the robot hits the wall more than wall_hit_limit times
        """

        if maze_env is None and maze_env_kwargs is None:
            raise ValueError("Either maze_env or maze_env_kwargs must be given")
        if maze_env is None:
            maze_env = mz.MazeEnv(**maze_env_kwargs)

        self.maze_env = maze_env
        self.max_stepper_steps = max_stepper_steps
        self.max_steps = max_steps
        self.epsilon_to_hit_subgoal = epsilon_to_hit_subgoal
        self.max_vel_in_subgoal = max_vel_in_subgoal
        self.rewards_config = rewards
        self.done_on_collision = done_on_collision
        self.normalize_observations = normalize_observations
        self.wall_hit_limit = wall_hit_limit

        if not maze_env.xy_in_obs:
            raise Exception("In order to train a navigator, xy_in_obs is required for the environment")

        # make sure:
        if not self.maze_env.done_on_collision == done_on_collision:
            print("WARNING: done_on_collision is different in mazeEnv and navigatorEnv, changing mazeEnv")
            self.maze_env.done_on_collision = done_on_collision
        # make sure:
        if not maze_env.hit_target_epsilon == epsilon_to_hit_subgoal:
            print("WARNING: epsilon_to_hit_subgoal is different in mazeEnv and navigatorEnv, changing mazeEnv")
            maze_env.hit_target_epsilon = epsilon_to_hit_subgoal
        # make sure:
        if not maze_env.max_goal_velocity == max_vel_in_subgoal:
            print("WARNING: max_vel_in_subgoal is different in mazeEnv and navigatorEnv, changing mazeEnv")
            maze_env.max_goal_velocity = max_vel_in_subgoal
        # make sure there is no requirement for heading angle at the at goal (success is only defined by distance):
        self.maze_env.target_heading_epsilon = np.inf

        self.visualize = False
        self.visualize_fps = 40

        if stepper_agent is None:
            stepper_agent = StepperAgent('TrainingNavigator/StepperAgents/StepperAntNoRotation.pt', 'auto')
        elif isinstance(stepper_agent, str):
            stepper_agent = StepperAgent(stepper_agent, 'auto')
        self.stepper_agent = stepper_agent

        # robot's current state
        self.robot_curr_obs = np.zeros(self.maze_env.observation_space.shape)
        self.curr_subgoal = np.zeros(2)
        self.target_goal = np.array(self.maze_env.workspace.goal_loc_tuple(), dtype=np.float32)

        # Action -> [radius, direction to subgoal (raidans)]
        self.action_space = Box(low=np.array([stepper_radius_range[0], -math.pi], dtype=np.float32),
                                high=np.array([stepper_radius_range[1], math.pi], dtype=np.float32),
                                shape=(2,))

        # Observation -> [ Agent_x, Agent_y,  Agent_Heading, Goal_x, Goal_y]
        self.observation_space = Box(-np.inf, np.inf, (5,))

        self.curr_step = 0
        self.wall_hit_count = 0
        self.total_stepper_steps = 0

    def reset(self, **maze_env_kwargs) -> np.ndarray:
        self.curr_step = 0
        self.robot_curr_obs = self.maze_env.reset(**maze_env_kwargs)
        self.wall_hit_count = 0

        nav_obs = self.get_nav_obs()
        return self.normalize_obs_if_needed(nav_obs)

    def step(self, action, visualize_subgoal=True):

        robot_xy = self.robot_curr_obs[0:2]

        # 2 elements of action are range and direction to the subgoal
        self.curr_subgoal = robot_xy + pol2cart(action[0:2])

        if visualize_subgoal:
            self.maze_env.set_subgoal_marker(self.curr_subgoal)
        else:
            self.maze_env.set_subgoal_marker(visible=False)

        _hit_maze = False

        for i in range(self.max_stepper_steps):
            if self.visualize:
                time.sleep(1. / float(self.visualize_fps))

            self.total_stepper_steps += 1

            # compute the (r, theta) from the *current* robot location to the subgoal in order to put it
            # in the observation to the stepper this is different then the value passed in the action
            # to this env because it changes in each iteration in the loop
            r_theta_to_subgoal = cart2pol(self.curr_subgoal - robot_xy)
            # we used subgoal-robot_loc and not the opposite just like in mazeEnv._get_observation

            # create observation for the stepper:
            stepper_obs = self.get_stepper_obs(r_theta_to_subgoal)

            stepper_action = self.stepper_agent.step(stepper_obs)

            # play robot step, reward is not required and is_done is determined using info
            self.robot_curr_obs, _, _, info = self.maze_env.step(stepper_action)
            robot_xy = self.robot_curr_obs[0:2]
            robot_velocity = np.sqrt(self.robot_curr_obs[3] ** 2 + self.robot_curr_obs[4] ** 2)

            # use info to check if finished (may override mazeEnv is_done in some cases):
            if info['success'] or info['fell'] or info['TimeLimit.truncated']:
                break
            if info['hit_maze']:
                _hit_maze = True
                if self.done_on_collision:
                    break
            # check if meet subgoal criteria:
            if np.linalg.norm(self.curr_subgoal - robot_xy) < self.epsilon_to_hit_subgoal \
                    and robot_velocity < self.max_vel_in_subgoal:
                break

        nav_observation = self.get_curr_nav_obs()

        nav_info = info
        nav_info['too_many_wallhits'] = False

        # determine nav reward and nav done according to info:
        nav_reward = 0
        nav_is_done = False
        if _hit_maze:
            self.wall_hit_count += 1
            nav_reward += self.rewards_config.collision
            too_many_walls_hit = 0 < self.wall_hit_limit < self.wall_hit_count
            nav_is_done = self.done_on_collision or too_many_walls_hit
            nav_info['too_many_wallhits'] = too_many_walls_hit
            nav_info['hit_maze'] = True
        elif info['fell']:
            nav_reward += self.rewards_config.fall
            nav_is_done = True
        elif info['success']:
            nav_reward += self.rewards_config.target_arrival
            nav_is_done = True
        else:
            nav_reward += self.rewards_config.idle

        nav_info['stepper_timeout'] = nav_info.pop('TimeLimit.truncated')
        self.curr_step += 1
        if self.curr_step >= self.max_steps:
            nav_info['TimeLimit.truncated'] = True
            nav_is_done = True
        else:
            nav_info['TimeLimit.truncated'] = False

        nav_observation = self.normalize_obs_if_needed(nav_observation)

        return nav_observation, nav_reward, nav_is_done, nav_info

    def visualize_mode(self, visualize: bool, fps: int = 40):
        """
        Change to (or change back from) visualize mode.
        In visualize mode the simulation is slowed down in order to visualize it in real-time
        :param visualize: weather to slow down the simulation to visualize
        :param fps: number of frames per second, actual fps may be inaccurate
        """
        self.visualize = visualize
        self.visualize_fps = fps

    def get_nav_obs(self):
        return np.concatenate([self.robot_curr_obs[0:2], [self.robot_curr_obs[8]], self.target_goal],
                              dtype=np.float32)

    def normalize_obs_if_needed(self, obs):
        norm_obs = obs.copy()
        if self.normalize_observations:
            # normalize x and y of robot and goal:
            maze_size_x, maze_size_y = self.maze_env.maze_size
            max_xy = np.array([maze_size_x, maze_size_y])
            norm_obs[0:2] = 2 * (norm_obs[0:2] / max_xy) - 1
            norm_obs[3:5] = 2 * (norm_obs[3:5] / max_xy) - 1

            # normalize rotation:
            norm_obs[2] = norm_obs[2] / np.pi

        return norm_obs

    def unormalize_obs_if_needed(self, obs):
        unorm_obs = obs.copy()
        if self.normalize_observations:
            maze_size_x, maze_size_y = self.maze_env.maze_size
            max_xy = np.array([maze_size_x, maze_size_y])
            unorm_obs[0:2] = (unorm_obs[0:2] + 1) * max_xy / 2
            unorm_obs[3:5] = (unorm_obs[3:5] + 1) * max_xy / 2

            unorm_obs[2] = unorm_obs[2] * np.pi

        return unorm_obs

    def get_curr_nav_obs(self):
        return np.concatenate([self.robot_curr_obs[0:2], [self.robot_curr_obs[8]], self.target_goal],
                              dtype=np.float32)

    def get_stepper_obs(self, r_theta_to_subgoal):
        obs = self.robot_curr_obs
        # update goal to new subgoal before transforming to stepper obs:
        obs[12:14] = r_theta_to_subgoal
        return transform_to_stepper_obs(obs, self.maze_env.get_joint_state_dim())


class MultiWorkspaceNavigatorEnv(NavigatorEnv):
    """
     Navigator Environment with multiple start and goal pairs, varying every episode
    """

    def __init__(self, workspace_list: list, repeat_failed_ws_prob=0, **navigator_kwargs):
        super(MultiWorkspaceNavigatorEnv, self).__init__(**navigator_kwargs)

        self.workspace_list = workspace_list
        self.repeat_failed_ws_prob = repeat_failed_ws_prob

        self.workspace_count = len(self.workspace_list)
        self.curr_ws_index = None

        self.is_last_failed = False

    def reset(self, workspace_idx: int = None, **kwargs):
        """
        Reset the environment to a new start-goal pair
        :param workspace_idx: idx of the start-goal pair, if none a random one is chosen
        :return: observation
        """
        if workspace_idx is None:
            if self.is_last_failed and np.random.rand() < self.repeat_failed_ws_prob:
                workspace_idx = self.curr_ws_index
            else:
                workspace_idx = np.random.randint(0, self.workspace_count)

        if workspace_idx >= self.workspace_count:
            raise ValueError("workspace index is out of range")

        self.curr_ws_index = workspace_idx

        self.maze_env.set_workspace(self.workspace_list[workspace_idx])
        self.target_goal = self.workspace_list[workspace_idx].goal_loc_tuple()

        return super(MultiWorkspaceNavigatorEnv, self).reset(**kwargs)

    def step(self, action, visualize_subgoal=True):
        """
        same as NavigatorEnv.step. only difference is that info contains the start-goal pair idx

        """
        obs, reward, is_done, info = super(MultiWorkspaceNavigatorEnv, self).step(action, visualize_subgoal)
        self.is_last_failed = not info['success']
        info['workspace_idx'] = self.curr_ws_index
        return obs, reward, is_done, info
