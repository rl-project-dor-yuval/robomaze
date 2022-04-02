import gym
from gym.spaces import Box
import MazeEnv.MazeEnv as mz
from TrainingNavigator.StepperAgent import StepperAgent
import numpy as np
import math
import time


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
    :param maze_env: mz.MazeEnv object - with observation space of 30d - with ant x,y location
    :param stepper_agent: stepper agent object
    :param max_stepper_steps: maximum steps allowed to the stepper towards subgoal.
    :param stepper_radius_range: defined by the reachable
    :param epsilon_to_hit_subgoal: minimal distance to subgoal to be considered as goal arrival

    NavigatorEnv is a wrapper around MazeEnv and a stepper agent so that an s here is a command to the stepper.
    Generally the navigator's purpose is given a certain Goal, generate a sequence of subgoals (Navigator actions)
    all the way to it, whereas the stepperAgent knows how to reach close subgoals.
    """

    def __init__(self,
                 maze_env: mz.MazeEnv,
                 stepper_agent=None,
                 max_stepper_steps=200,
                 stepper_radius_range=(0.9, 3.),
                 epsilon_to_hit_subgoal=0.7):

        if not maze_env.xy_in_obs:
            raise Exception("In order to train a navigator, xy_in_obs is required for the environment")

        self.maze_env = maze_env
        self.max_stepper_steps = max_stepper_steps
        self.epsilon_to_hit_subgoal = epsilon_to_hit_subgoal
        # make sure:
        maze_env.hit_target_epsilon = epsilon_to_hit_subgoal

        self.visualize = False
        self.visualize_fps = 40

        if stepper_agent is None:
            stepper_agent = StepperAgent('StepperAgent.pt', 'auto')
        self.stepper_agent = stepper_agent

        # Ant's current state
        self.ant_curr_obs = np.zeros(30)
        self.curr_subgoal = np.zeros(2)
        self.target_goal = np.array(self.maze_env._target_loc[0:2], dtype=np.float32)

        # Action -> [radius, azimuth (in degrees)]
        self.action_space = Box(low=np.array([stepper_radius_range[0], -math.pi], dtype=np.float32),
                                high=np.array([stepper_radius_range[1], math.pi], dtype=np.float32),
                                shape=(2,))
        # Observation -> [ Agent_x, Agent_y, Target_x, Target_y]
        self.observation_space = Box(-np.inf, np.inf, (4,))

    def reset(self):
        self.ant_curr_obs = self.maze_env.reset()
        return np.concatenate([self.ant_curr_obs[0:2], self.target_goal], dtype=np.float32)

    def step(self, action, visualize_subgoal=True):
        ant_xy = self.ant_curr_obs[0:2]
        # 2 first elements of action are range and direction to the subgoal
        self.curr_subgoal = ant_xy + pol2cart(action[0:2])

        if visualize_subgoal:
            self.maze_env.set_subgoal_marker(self.curr_subgoal)
        else:
            self.maze_env.set_subgoal_marker(visible=False)

        nav_reward = 0

        for i in range(self.max_stepper_steps):
            if self.visualize:
                time.sleep(1./float(self.visualize_fps))

            # compute the (r, theta) from the *current* ant location to the subgoal in order to put it
            # in the observation to the stepper this is different then the value passed in the action
            # to this env because it changes in each iteration in the loop
            r_theta_to_subgoal = cart2pol(self.curr_subgoal - ant_xy)
            # we used subgoal-ant_loc and not the opposite just like in mazeEnv._get_observation

            # stepper agent doesn't need x and y of the ant
            stepper_obs = self.ant_curr_obs[2:]
            # update r and theta for the subgoal (the environment returns for the main goal)
            stepper_obs[26:28] = r_theta_to_subgoal
            stepper_action = self.stepper_agent.step(stepper_obs)

            self.ant_curr_obs, reward, is_done, info = self.maze_env.step(stepper_action)
            ant_xy = self.ant_curr_obs[0:2]

            nav_reward += reward

            # check main goal arrival
            if is_done:
                break

            # check if close enough to subgoal:
            if np.linalg.norm(self.curr_subgoal - ant_xy) < self.epsilon_to_hit_subgoal:
                break

        nav_observation = np.concatenate([ant_xy, self.target_goal])
        nav_info = info

        return nav_observation, nav_reward, is_done, nav_info

    def visualize_mode(self, visualize: bool, fps: int = 40):
        """
        Change to (or change back from) visualize mode.
        In visualize mode the simulation is slowed down in order to visualize it in real-time
        :param visualize: weather to slow down the simulation to visualize
        :param fps: number of frames per second, actual fps may be inaccurate
        """
        self.visualize = visualize
        self.visualize_fps = fps


class MultiStartgoalNavigatorEnv(NavigatorEnv):
    """
     Navigator Environment with multiple start and goal pairs, varying every episode
    """
    def __init__(self, target_goal_pairs: np.ndarray, **navigator_kwargs):
        super(MultiStartgoalNavigatorEnv, self).__init__(**navigator_kwargs)

        self.target_goal_pairs = target_goal_pairs
        self.start_goal_pairs_count = len(self.target_goal_pairs)

    def reset(self, start_goal_pair_idx: int = None):
        """
        Reset the environment to a new start-goal pair
        :param start_goal_pair_idx: idx of the start-goal pair, if none a random one is chosen
        :return: observation
        """
        if start_goal_pair_idx is None:
            start_goal_pair_idx = np.random.randint(0, self.start_goal_pairs_count)
        if start_goal_pair_idx >= self.start_goal_pairs_count:
            raise ValueError("start_goal_pair_idx is out of range")

        self.maze_env.set_target_loc(self.target_goal_pairs[start_goal_pair_idx][1])
        self.maze_env.set_start_loc(self.target_goal_pairs[start_goal_pair_idx][0])

        return super(MultiStartgoalNavigatorEnv, self).reset()

    def step(self, action, visualize_subgoal=True):
        """
        same as NavigatorEnv.step. only difference is that info contains the start-goal pair idx

        """
        obs, reward, is_done, info = super(MultiStartgoalNavigatorEnv, self).step(action, visualize_subgoal)
        info['start_goal_pair_idx'] = self.maze_env.start_goal_pair_idx
        return obs, reward, is_done, info
