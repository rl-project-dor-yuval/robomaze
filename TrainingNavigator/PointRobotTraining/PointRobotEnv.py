import gym
from gym.spaces import Box
from TrainingNavigator.StepperAgent import StepperAgent

import numpy as np
import math
import matplotlib.pyplot as plt
import time
from MazeEnv.EnvAttributes import Rewards
from TrainingNavigator.NavigatorEnv import pol2cart
from TrainingNavigator.Utils import tf_top_left_to_bottom_left


class PointRobotEnv(gym.Env):
    """
    The following environment plays a point robot in the given maze.
    :param maze: The maze to play in.
    :param max_steps: The maximum number of steps to take before the episode is terminated.
    :param rewards: The rewards to use for the environment.
    :param radius_range: The range of the radius of the point robot.
    :param done_on_collision: Whether or not the episode is done when the point robot collides with the wall.
    :param epsilon_to_hit_subgoal: The epsilon value to be a success.
    :param start_goal_pairs: The start and goal pairs to use for the environment.
    :param maze_map: The map of the maze.
    """

    def __init__(self,
                 max_steps=100,
                 rewards=Rewards(),
                 radius_range=(0.6, 2.5),
                 done_on_collision=True,
                 epsilon_to_hit_subgoal=0.8,
                 start_goal_pairs: np.ndarray = None,
                 maze_map: np.ndarray = None,
                 visualize=False,
                 ):
        self.max_steps = max_steps
        self.rewards = rewards
        self.radius_range = radius_range
        self.done_on_collision = done_on_collision
        self.epsilon_to_hit_subgoal = epsilon_to_hit_subgoal
        self.maze_map = maze_map

        # RL params

        # Action -> [radius, azimuth (in radian)]
        self.action_space = Box(low=np.array([radius_range[0], -math.pi], dtype=np.float32),
                                high=np.array([radius_range[1], math.pi], dtype=np.float32),
                                shape=(2,))

        # Observation -> [ Agent_x, Agent_y, Target_x, Target_y]
        self.observation_space = Box(-np.inf, np.inf, (4,))

        self.curr_step = 0
        self.visualize = visualize
        if self.visualize:
            self.fig = plt.figure(figsize=(10, 10))
            self.scale_const = 10

        # multi-start-goal environment params
        self.start_goal_pairs = start_goal_pairs
        self.start_goal_pairs_count = len(self.start_goal_pairs)
        self.curr_startgoal_pair_idx = None

    def reset(self, start_goal_pair_idx=None):
        """
        reset the environment to new start-goal pair
        :param start_goal_pair_idx: index of start-goal pair
        """
        if start_goal_pair_idx is None:
            start_goal_pair_idx = np.random.randint(0, self.start_goal_pairs_count)
        if start_goal_pair_idx >= self.start_goal_pairs_count:
            raise ValueError("start_goal_pair_idx is out of range")

        self.curr_startgoal_pair_idx = start_goal_pair_idx

        # set current start and goal position
        # self.current_target_loc = tf_top_left_to_bottom_left(self.start_goal_pairs[start_goal_pair_idx][1] * self.scale_const,\
        #                                                      self.maze_map.shape[0])
        self.current_target_loc = self.start_goal_pairs[start_goal_pair_idx][1]  # / self.scale_const
        # self.current_start_loc = tf_top_left_to_bottom_left(self.start_goal_pairs[start_goal_pair_idx][0] * self.scale_const,\
        #                                                     self.maze_map.shape[0])
        self.current_start_loc = self.start_goal_pairs[start_goal_pair_idx][0]  # / self.scale_const

        self.curr_pnt_loc = self.current_start_loc

        self.curr_step = 0

        if self.visualize:
            self.visualize_env()

    def step(self, action):
        assert action in self.action_space, "Action is not in action space"

        pnt_xy = self.curr_pnt_loc
        # calculate next subgoal towards the goal
        next_pnt_xy = pnt_xy + pol2cart(action)

        pnt_observation = np.concatenate([next_pnt_xy, self.current_target_loc])
        pnt_reward = 0
        pnt_info = {}
        pnt_isdone = False

        _hit_maze, _hit_goal = self._check_collision(next_pnt_xy)
        if _hit_maze:
            # collision
            if self.done_on_collision:
                pnt_isdone = True
            pnt_reward += self.rewards.collision
        elif _hit_goal:
            # goal
            pnt_isdone = True
            pnt_reward += self.rewards.goa
        else:
            # idle
            pnt_reward += self.rewards.idle

        self.curr_step += 1
        if self.curr_step >= self.max_steps:
            pnt_info['pnt_timeout'] = True
            pnt_isdone = True
        else:
            pnt_info['pnt_timeout'] = False

        self.curr_pnt_loc = next_pnt_xy
        if self.visualize:
            self.visualize_env()

        return pnt_observation, pnt_reward, pnt_isdone, pnt_info

    def _check_collision(self, pnt_xy, PATIENCE_VAL=2):
        """
        Check if the point is a collision, that is, if the point is in radius R_PATIENCE_VAL of any obstacle or
        goal.
        """
        # check if the ant is in PATIENCE_VAL away from the goal before or after
        x, y = pnt_xy[0], pnt_xy[1]
        left, right = int(x * self.scale_const - PATIENCE_VAL), int(x * self.scale_const + PATIENCE_VAL)
        down, up = int(y * self.scale_const - PATIENCE_VAL), int(y * self.scale_const + PATIENCE_VAL)

        for i in range(left, right):
            for j in range(down, up):
                # for r in np.linspace(0, R_PATIENCE_VAL, 8):
                #     for t in np.linspace(-np.pi, np.pi, 8):
                #         x, y = pol2cart(np.array([r, t]))
                #         x, y = int(x * self.scale_const + pnt_xy[0]),\
                #                int(y * self.scale_const + pnt_xy[1])

                # # check if i, j inside maze map size
                # if i < 0 or i >= self.maze_map.shape[0] or j < 0 or j >= self.maze_map.shape[1]:
                #     continue

                if np.linalg.norm(np.array([j, i]) - self.current_target_loc) < self.epsilon_to_hit_subgoal:
                    hit_maze = False
                    hit_goal = True
                    return hit_maze, hit_goal

                if self.maze_map[j, i] == 1 or \
                        (i < 0 or i >= self.maze_map.shape[0] or j < 0 or j >= self.maze_map.shape[1]):
                    self.maze_map[j, i] == 0.1  # mark visited for debug
                    hit_maze = True
                    hit_goal = False
                    return hit_maze, hit_goal

                self.maze_map[j, i] = 0.5

        hit_maze = False
        return hit_maze, False

    def visualize_env(self):

        axe = self.fig.gca()

        # plot start and goal
        if self.curr_step == 0:
            self.fig.clf()
            axe = self.fig.gca()

        axe.imshow(self.maze_map, cmap='gray')
        axe.plot(self.current_start_loc[0] * self.scale_const, self.current_start_loc[1] * self.scale_const,
                 'go', markersize=20)
        axe.plot(self.current_target_loc[0] * self.scale_const, self.current_target_loc[1] * self.scale_const,
                 'bo', markersize=20)
        axe.set_title(f"Step: {self.curr_step}")

        # clean previous point loc and plot current point loc
        axe.plot(self.curr_pnt_loc[0] * self.scale_const, self.curr_pnt_loc[1] * self.scale_const,
                 'wx', markersize=10)

        # ----  debug -----

        # for i in range(200):
        #     # self.fig.gca()
        #     # start
        #     axe.plot(self.start_goal_pairs[i, 0, 1]*10, self.start_goal_pairs[i, 0, 0]*10, 'go', markersize=20)
        #     # goal
        #     axe.plot(self.start_goal_pairs[i, 1, 1]*10, self.start_goal_pairs[i, 1, 0]*10, 'bo', markersize=20)

        plt.show(block=False)
        plt.pause(0.01)
