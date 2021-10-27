"""
 A wrapper around MazeEnv and a stepper agent
"""
import gym
from gym.spaces import Box
import MazeEnv.MazeEnv as mz
from StepperAgent import StepperAgent
import numpy as np
import math
from Utils import cart2pol, pol2cart


class NavigatorEnv(gym.Env):
    def __init__(self, maze_env: mz.MazeEnv,
                 stepper_agent=None,
                 max_stepper_steps=200,
                 stepper_radius_range=(0.7, 3),
                 epsilon_to_hit_subgoal=0.5):
        self.maze_env = maze_env
        self.max_stepper_steps = max_stepper_steps
        self.epsilon_to_hit_subgoal = epsilon_to_hit_subgoal

        if stepper_agent is None:
            stepper_agent = StepperAgent('StepperAgent.py', 'auto')
        self.stepper_agent = stepper_agent

        self.ant_curr_obs = np.zeroes(30)
        self.curr_subgoal = np.zeroes(2)
        self.target_goal = np.array(self.maze_env._target_loc[0:2])

        self.action_space = Box(low=(stepper_radius_range[0], -math.pi),
                                high=(stepper_radius_range[1], math.pi))

        self.observation_space = Box(-np.inf, np.inf, (4,))

    def reset(self):
        self.ant_curr_obs = self.maze_env.reset()
        return np.concatenate([self.ant_curr_obs[0:2], self.target_goal])

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

            # stepper agent doesn't need x and y of the ant
            stepper_obs = self.ant_curr_obs[2:]
            stepper_obs[26:28] = r_theta_to_subgoal
            stepper_action = self.stepper_agent(stepper_obs)

            self.ant_curr_obs, reward, is_done, info = self.maze_env.step(stepper_action)
            ant_xy = self.ant_curr_obs[0:2]

            nav_reward += reward

            if is_done:
                break

            # check if close enough to subgoal:
            if np.linalg.norm(self.curr_subgoal - ant_xy) < self.epsilon_to_hit_subgoal:
                break



        nav_observation = np.concatenate([ant_xy, self.target_goal])
        nav_info = info

        return nav_observation, nav_reward, is_done, nav_info
