import numpy as np
from gym.spaces import Box

from MazeEnv.MultWorkspaceMazeEnv import MultiWorkspaceMazeEnv

"""
MultiWorkspaceMazeEnv is almost good enough to be a StepperEnv, but lately it has become to complicated to configure
it to match the requirements of both stepper and navigator, therefore we fix a stepper environment here, which is
similar to the MultiWorkspaceMazeEnv, but with small fixes done through inheritance.

Stepper observation space is derived from the observation space of the maze environment:
[
 0   -   robot_z;
 1:3 - robot_velocities r, theta, z where we rotate vx and vy to be relative to stepper coordinate system;
 4:6 - robot orientations yaw, pitch, roll, where yaw is rotated to be relative to goal (Stepper coordinate system);
 7:9 - robot angular velocities
 10  - distance from goal
 n   - robot joint states
 n   - robot joint velocities
] 
"""

class StepperEnv(MultiWorkspaceMazeEnv):
    def __init__(self, **mw_maze_env_kwargs):
        mw_maze_env_kwargs['xy_in_obs'] = True
        super().__init__(**mw_maze_env_kwargs)

        obs_space_size = 11 + self._robot.get_joint_state_dim()
        self.observation_space = Box(-np.inf, np.inf, (obs_space_size,))

    def _transform_observation(self, observation):
        """
        transform observation from MazeEnv observation space to Stepper observation space
        :param observation: original observation recived from mazeEnv
        :return: stepper observation
        """

        direction_to_goal = observation[13]
        # we create a rotation matrix that rotates the direction to goal to zero, so we rotate by -direction_to_goal
        rotation_matrix = np.array([[np.cos(-direction_to_goal), -np.sin(-direction_to_goal)],
                                    [np.sin(-direction_to_goal), np.cos(-direction_to_goal)]])

        new_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        new_obs[0] = observation[2]  # z coord
        new_obs[1:3] = np.dot(rotation_matrix, observation[3:5])  # vx, vy -> vr, v_orth_to_r
        new_obs[3:6] = observation[5:8]  # vz, roll, pitch are the same
        # again, we rotate it  by -direction_to_goal angles, this can be computed using the transformation matrix
        # instead in maze env one day:
        new_obs[6] = self.compute_signed_rotation_diff(observation[8] - direction_to_goal)  # yaw -> new_yaw
        new_obs[7:10] = observation[9:12]  # v_yaw, v_pitch, v_roll are the same
        new_obs[10] = observation[12]  # distance from goal
        new_obs[11:] = observation[15:]  # joint states and velocities

        return new_obs

    def reset(self, *args, **kwargs):
        observation = super().reset(*args, **kwargs)
        return self._transform_observation(observation)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return self._transform_observation(observation), reward, done, info

