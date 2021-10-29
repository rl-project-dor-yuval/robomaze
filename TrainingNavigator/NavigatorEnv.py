"""
 A wrapper around MazeEnv and a stepper agent
"""
import gym
from gym.spaces import Box
import MazeEnv.MazeEnv as mz
from TrainingNavigator.StepperAgent import StepperAgent
import numpy as np
import math
from TrainingNavigator.Utils import cart2pol, pol2cart


class NavigatorEnv(gym.Env):
    """
    :param maze_env: mz.MazeEnv object - with observation space of 28d - without ant x,y location
    :param stepper_agent: stepper agent object
    :param max_stepper_steps: maximum steps allowed to the stepper towards subgoal.
    :param stepper_radius_range: defined by the reachable
    :param epsilon_to_hit_subgoal: minimal distance to subgoal to be considered as goal arrival

    The NavigatorEnv class will learn how to navigate the stepper agent in order to solve
    a certain maze. Generally the navigator's purpose is given a certain Goal, generate a sequence
    of subgoals (Navigator actions) all the way to it, whereas the stepperAgent knows how to reach close subgoals.
    """

    def __init__(self,
                 maze_env: mz.MazeEnv,
                 stepper_agent=None,
                 max_stepper_steps=200,
                 stepper_radius_range=(0.7, 3.),
                 epsilon_to_hit_subgoal=0.5):

        if not maze_env.xy_in_obs:
            raise Exception(" Navigator's env has to get agent's position in env - please provide proper env object")
            exit()
        self.maze_env = maze_env
        self.max_stepper_steps = max_stepper_steps
        self.epsilon_to_hit_subgoal = epsilon_to_hit_subgoal

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

    def step(self, action):
        ant_xy = self.ant_curr_obs[0:2]
        # 2 first elements of action are range and direction to the subgoal
        self.curr_subgoal = ant_xy + pol2cart(action[0:2])

        nav_reward = 0

        for i in range(self.max_stepper_steps):
            # compute the (r, theta) from the *current* ant location to the subgoal in order to put it
            # in the observation to the stepper this is different then the value passed in the action
            # to this env because it changes in each iteration in the loop
            r_theta_to_subgoal = cart2pol(self.curr_subgoal - ant_xy)
            # we used subgoal-ant_loc and not the opposite just like in mazeEnv._get_observation

            # stepper agent doesn't need x and y of the ant so make sure
            stepper_obs = self.ant_curr_obs[2:]
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
                is_done = True
                break

        nav_observation = np.concatenate([ant_xy, self.target_goal])
        nav_info = info

        return nav_observation, nav_reward, is_done, nav_info
