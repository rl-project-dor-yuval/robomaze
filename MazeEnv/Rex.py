from MazeEnv.RobotBase import RobotBase, scale
import numpy as np


START_HEIGHT = 1.1
JOINTS_INDICES = np.arange(0, 12)


class Rex(RobotBase):
    """
    Rex is a 12 joint robot with 3 motors for each leg: Shoulder- degrees in rad - left & right spin of leg
                                                        Leg- degrees in rad - back & forth spin of leg
                                                        Foot- degrees in rad
    - every entry accept any value and treat it as radians.
    0-2 Rear Left leg
    3-5 Rear Right leg
    6-8 Front left leg
    9-11 Front right leg
    """

    def __init__(self, pybullet_client, position2d, heading):
        position3d = np.concatenate((position2d, (START_HEIGHT,)))
        super().__init__(pybullet_client, position3d, heading, "quadrupedal.urdf", scale_urdf=2.5)

        # color robot:
        for link in range(self._pclient.getNumJoints(self.uid)):
            self._pclient.changeVisualShape(self.uid, linkIndex=link, rgbaColor=[0.4, 0.4, 0.4, 1])
        self._pclient.changeVisualShape(self.uid, linkIndex=-1, rgbaColor=[0.2, 0.2, 0.2, 1])

    def _reset_joints(self, noisy_state):

        for joint in JOINTS_INDICES:
            state_ = 0
            velocity = 0
            if noisy_state:
                if joint in [0, 3, 6, 9]:
                    # shoulder joints, should move less
                    state_ += np.random.uniform(-np.pi/16, np.pi/16)
                    velocity += np.random.uniform(-0.05, 0.05)
                else:
                    state_ += np.random.uniform(-np.pi/8, np.pi/8)
                    velocity += np.random.uniform(-0.1, 0.1)
            self._pclient.resetJointState(self.uid, joint, state_, velocity)

    def action(self, in_action: np.array):
        action = np.array(in_action, dtype=np.float32)

        mode = self._pclient.TORQUE_CONTROL
        self._pclient.setJointMotorControlArray(self.uid, JOINTS_INDICES, mode, forces=action * 40)

    def get_joint_state(self):
        """
        :return: 24d vector

        12 first values are joints position
        12 next values are joint velocity
        """
        joint_states_tuple = self._pclient.getJointStates(self.uid, JOINTS_INDICES)
        positions = np.array([j_state[0] for j_state in joint_states_tuple])
        velocities = np.array([j_state[1] for j_state in joint_states_tuple])

        # positions[::2] = scale(positions[::2], SHOULDER_HIGH, SHOULDER_LOW, 1, -1, )
        # positions[1] = -scale(positions[1], ANKLE_1_4_HIGH, ANKLE_1_4_LOW, 1, -1)
        # positions[7] = -scale(positions[7], ANKLE_1_4_HIGH, ANKLE_1_4_LOW, 1, -1)
        # positions[5] = scale(positions[5], ANKLE_2_3_HIGH, ANKLE_2_3_LOW, 1, -1)
        # positions[3] = scale(positions[3], ANKLE_2_3_HIGH, ANKLE_2_3_LOW, 1, -1)

        return np.concatenate((positions, velocities))

    def get_action_dim(self):
        return 12

    def get_joint_state_dim(self):
        return 24
