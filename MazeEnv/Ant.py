from MazeEnv.RobotBase import RobotBase, scale
import numpy as np


# the indices of the joints in the built model
# ANKLE_IDX = np.array([3, 8, 13, 18])
# SHOULDER_IDX = np.array([1, 6, 11, 16])
JOINTS_INDICES = np.array([1, 3, 6, 8, 11, 13, 16, 18])
# Practical joints motors ranges
SHOULDER_HIGH = 0.698
SHOULDER_LOW = -0.698
ANKLE_1_4_HIGH = 1.745
ANKLE_1_4_LOW = 0.523
ANKLE_2_3_HIGH = -0.523
ANKLE_2_3_LOW = -1.745
INIT_JOINT_STATES = [0, (ANKLE_1_4_HIGH + ANKLE_1_4_LOW) / 2,
                     0, (ANKLE_2_3_HIGH + ANKLE_2_3_LOW) / 2,
                     0, (ANKLE_2_3_HIGH + ANKLE_2_3_LOW) / 2,
                     0, (ANKLE_1_4_HIGH + ANKLE_1_4_LOW) / 2]


class Ant(RobotBase):
    def __init__(self, pybullet_client, position3d, heading):
        super().__init__(pybullet_client, position3d, heading, "ant.xml", scale_urdf=None)

        # color ant:
        for link in range(self._pclient.getNumJoints(self.uid)):
            self._pclient.changeVisualShape(self.uid, linkIndex=link, rgbaColor=[0.4, 0.4, 0.4, 1])
        self._pclient.changeVisualShape(self.uid, linkIndex=-1, rgbaColor=[0.2, 0.2, 0.2, 1])

    def _reset_joints(self, noisy_state):
        for joint, state in zip(JOINTS_INDICES, INIT_JOINT_STATES):
            state_ = state
            velocity = 0
            if noisy_state:
                state_ += np.random.uniform(-1, 1)
                velocity += np.random.uniform(-0.5, 0.5)
            self._pclient.resetJointState(self.uid, joint, state_, velocity)

    def action(self, in_action: np.array):
        action = np.array(in_action, dtype=np.float32)

        mode = self._pclient.TORQUE_CONTROL
        self._pclient.setJointMotorControlArray(self.uid, JOINTS_INDICES, mode, forces=action * 1500)

    def get_joint_state(self):
        """
        :return: 16d vector

        8 first values are joints position
        8 next values are joint velocity
        """
        joint_states_tuple = self._pclient.getJointStates(self.uid, JOINTS_INDICES)
        positions = np.array([j_state[0] for j_state in joint_states_tuple])
        velocities = np.array([j_state[1] for j_state in joint_states_tuple])

        positions[::2] = scale(positions[::2], SHOULDER_HIGH, SHOULDER_LOW, 1, -1, )
        positions[1] = -scale(positions[1], ANKLE_1_4_HIGH, ANKLE_1_4_LOW, 1, -1)
        positions[7] = -scale(positions[7], ANKLE_1_4_HIGH, ANKLE_1_4_LOW, 1, -1)
        positions[5] = scale(positions[5], ANKLE_2_3_HIGH, ANKLE_2_3_LOW, 1, -1)
        positions[3] = scale(positions[3], ANKLE_2_3_HIGH, ANKLE_2_3_LOW, 1, -1)

        return np.concatenate((positions, velocities))

    def get_action_dim(self):
        return 8

    def _get_joint_state_dim(self):
        return 16
