import numpy as np
from gym.spaces import Box

from MazeEnv.MultWorkspaceMazeEnv import MultiWorkspaceMazeEnv
from Training.TransformObservation import transform_to_stepper_obs, OBS_SPACE_SIZE_NO_JOINT_STATES

"""
MultiWorkspaceMazeEnv is almost good enough to be a StepperEnv, but lately it has become to complicated to configure
it to match the requirements of both stepper and navigator, therefore we fix a stepper environment here, which is
similar to the MultiWorkspaceMazeEnv, but with small fixes done through inheritance.

Stepper observation space is derived from the observation space of the maze environment:
[
 0   -   robot_z;
 1:3 - robot_velocities r, theta, z where we rotate vx and vy to be relative to stepper coordinate system;
 4:6 - robot orientations yaw, pitch, roll, where yaw is rotated to be relative to goal (Stepper coordinate system);
 7:9 - robot angular velocities;
 10  - distance from goal
 11  - heading diff from desired heading at goal
 n   - robot joint states
 n   - robot joint velocities
] 
"""


class StepperEnv(MultiWorkspaceMazeEnv):
    def __init__(self, **mw_maze_env_kwargs):
        # since episodes are shorter for stepper, we can allow ourselves to invest more resources in video:
        self.recording_video_size = (350, 350)
        self.video_skip_frames = 2

        super().__init__(**mw_maze_env_kwargs)

        # better view for stepper:
        self.set_view(-52.5, 1.3, 1.5)

        obs_space_size = OBS_SPACE_SIZE_NO_JOINT_STATES + self._robot.get_joint_state_dim()
        self.observation_space = Box(-np.inf, np.inf, (obs_space_size,))

    def reset(self, *args, **kwargs):
        observation = super().reset(*args, **kwargs)
        return transform_to_stepper_obs(observation, self._robot.get_joint_state_dim())

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return transform_to_stepper_obs(observation, self._robot.get_joint_state_dim()), reward, done, info

