"""
 A wrapper around MazeEnv and a stepper agent
"""
import gym

import MazeEnv.MazeEnv as mz
from StepperAgent import StepperAgent
import numpy as np


class NavigatorAgent(gym.Env):
    def __init__(self, maze_env: mz.MazeEnv, stepper_agent=None, max_stepper_steps=200):
        self.maze_env = maze_env

        if stepper_agent is None:
            stepper_agent = StepperAgent('StepperAgent.py', 'auto')
        self.stepper_agent = stepper_agent

        self.ant_curr_obs = np.zeroes(30)
        self.curr_subgoal = None

        self.target_goal = np.array(self.maze_env._target_loc[0:2])

        # self.action_space = TODO complete

    def step(self, action):
        ant_xy = self.ant_curr_obs[0:2]
        #
        # Todo Compute curr_subgoal
        # self.curr_subgoal =

        reward = 0

        for i in range(self.max_stepper_steps):
            # r_and_theta_to_subgoal = TODO complete
            # stepper_obs[28:30] = r_and_theta_to_subgoal

            # stepper agent doesn't need x and y of the ant
            stepper_obs = self.ant_curr_obs[2:]
            stepper_action = self.stepper_agent(stepper_obs)

            self.ant_curr_obs, reward, is_done, info = self.maze_env.step(stepper_action)

            # if TODO check if close enough to current subgoal
                # break and more

            if is_done:
                # TODO finished

        nav_observation = np.concatenate([self.ant_curr_obs[0:2], self.target_goal])
        nav_info = None
        nav_reward = reward