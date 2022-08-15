import os.path
from MazeEnv.RobotBase import RobotBase
import numpy as np

START_HEIGHT = 1.2
JOINTS_INDICES = np.arange(0, 12)
POSITION_LIMITS = np.ones(12) * 2.5  # just like in the URDF
TORQUE_LIMITS = np.array([2, 2, 2, 10, 10, 10, 2, 2, 2, 10, 10, 10])


class Bipedal(RobotBase):
    """
    Bipedal robot is a 12 joint robot with 6 DOF for each leg:
    - hip_yaw/roll/pitch
    - knee_yaw/roll/pitch

    """

    _joint_name_to_id = {}

    def __init__(self, pybullet_client, position2d, heading):
        position3d = np.concatenate((position2d, (START_HEIGHT,)))
        super().__init__(pybullet_client, position3d, heading, "bipedal.urdf", scale_urdf=2)

        # color robot:
        for link in range(self._pclient.getNumJoints(self.uid)):
            self._pclient.changeVisualShape(self.uid, linkIndex=link, rgbaColor=[0.4, 0.4, 0.4, 1])
        self._pclient.changeVisualShape(self.uid, linkIndex=-1, rgbaColor=[0.2, 0.2, 0.2, 1])

    def _setup_robot(self):
        self._joint_name_to_id = {}
        self._build_joint_name2id_dict()

    def _build_joint_name2id_dict(self):
        self.num_joints = self._pclient.getNumJoints(self.uid)
        for i in range(self.num_joints):
            joint_info = self._pclient.getJointInfo(self.uid, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        print("Joint name to id dict:", self._joint_name_to_id)

    def _reset_joints(self, noisy_state):

        for j_id in range(self.num_joints):
            state_ = 0
            velocity = 0
            if noisy_state:
                if j_id in [0, 1, 2, 6, 7, 8]:
                # hips joints, should move less
                    state_ += np.random.uniform(-np.pi/32, np.pi/32)
                    velocity += np.random.uniform(-0.05, 0.05)
                else:
                    state_ += np.random.uniform(-np.pi/8, np.pi/8)
                    velocity += np.random.uniform(-0.1, 0.1)

            self._pclient.resetJointState(self.uid, j_id, state_, velocity)

    def action(self, in_action: np.array):
        # action = np.array(TORQUE_LIMITS * in_action, dtype=np.float32)
        # action = np.clip(action, -1 * TORQUE_LIMITS, TORQUE_LIMITS)
        #
        # mode = self._pclient.TORQUE_CONTROL
        # self._pclient.setJointMotorControlArray(self.uid, JOINTS_INDICES, mode, forces=action)

        action = np.array(in_action, dtype=np.float32) * POSITION_LIMITS
        mode = self._pclient.POSITION_CONTROL
        self._pclient.setJointMotorControlArray(self.uid, JOINTS_INDICES, mode, action,
                                                forces=TORQUE_LIMITS)

    def get_joint_state(self):
        """
        :return: 24d vector

        12 first values are joints position
        12 next values are joint velocity
        """
        joint_states_tuple = self._pclient.getJointStates(self.uid, JOINTS_INDICES)
        positions = np.array([j_state[0] for j_state in joint_states_tuple])
        velocities = np.array([j_state[1] for j_state in joint_states_tuple])

        return np.concatenate((positions, velocities))

    def get_action_dim(self):
        return 12

    def get_joint_state_dim(self):
        return 24
