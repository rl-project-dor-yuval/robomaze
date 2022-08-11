""" this method was take out of StepperEnv because it is used by navigator as well """

import numpy as np
from MazeEnv.MazeEnv import MazeEnv


def transform_to_stepper_obs(observation, robot_joint_state_dim):
    """
    transform observation from MazeEnv observation space to Stepper observation space including xy_in_obs
    :param observation: original observation recived from mazeEnv
    :return: stepper observation
    """

    direction_to_goal = observation[13]
    # we create a rotation matrix that rotates the direction to goal to zero, so we rotate by -direction_to_goal
    rotation_matrix = np.array([[np.cos(-direction_to_goal), -np.sin(-direction_to_goal)],
                                [np.sin(-direction_to_goal), np.cos(-direction_to_goal)]])

    new_obs = np.zeros(11 + robot_joint_state_dim, dtype=np.float32)

    new_obs[0] = observation[2]  # z coord
    new_obs[1:3] = np.dot(rotation_matrix, observation[3:5])  # vx, vy -> vr, v_orth_to_r
    new_obs[3:6] = observation[5:8]  # vz, roll, pitch are the same
    # again, we rotate it  by -direction_to_goal angles, this can be computed using the transformation matrix
    # instead in maze env one day:
    new_obs[6] = MazeEnv.compute_signed_rotation_diff(observation[8] - direction_to_goal)  # yaw -> new_yaw
    new_obs[7:10] = observation[9:12]  # v_yaw, v_pitch, v_roll are the same
    new_obs[10] = observation[12]  # distance from goal
    new_obs[11:] = observation[15:]  # joint states and velocities

    return new_obs