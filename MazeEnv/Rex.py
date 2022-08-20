import os.path

from MazeEnv.RobotBase import RobotBase, scale
import numpy as np

START_HEIGHT = 1.1
JOINTS_INDICES = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]

LOWER_JOINT_LIMITS = np.array(
    [-1.0472, -0.523599, -2.77507, -0.872665, -0.523599, -2.77507, -1.0472, -0.523599, -2.77507,
     -0.872665, -0.523599, -2.77507])
UPPER_JOINT_LIMITS = np.array(
    [0.872665, 3.92699, -0.610865, 1.0472, 3.92699, -0.610865, 0.872665, 3.92699, -0.610865, 1.0472,
     3.92699, -0.610865])
TORQUE_LIMITS = np.ones(12) * 100

STAND_MOTOR_ANGLES = np.array([-10, 30, -75,
                               10, 30, -75,
                               -10, 50, -75,
                               10, 50, -75]) * np.pi / 180

JOINTS_NAMES = [
    "FR_hip_motor_2_chassis_joint",
    "FR_upper_leg_2_hip_motor_joint",
    "FR_lower_leg_2_upper_leg_joint",
    "FL_hip_motor_2_chassis_joint",
    "FL_upper_leg_2_hip_motor_joint",
    "FL_lower_leg_2_upper_leg_joint",
    "RR_hip_motor_2_chassis_joint",
    "RR_upper_leg_2_hip_motor_joint",
    "RR_lower_leg_2_upper_leg_joint",
    "RL_hip_motor_2_chassis_joint",
    "RL_upper_leg_2_hip_motor_joint",
    "RL_lower_leg_2_upper_leg_joint",
]


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

    _joint_name_to_id = {}

    def __init__(self, pybullet_client, position2d, heading):
        position3d = np.concatenate((position2d, (START_HEIGHT,)))
        super().__init__(pybullet_client, position3d, heading, os.path.join("laikago_urdf", "laikago.urdf"),
                         scale_urdf=2)

        # color robot:
        for link in range(self._pclient.getNumJoints(self.uid)):
            self._pclient.changeVisualShape(self.uid, linkIndex=link, rgbaColor=[0.4, 0.4, 0.4, 1])
        self._pclient.changeVisualShape(self.uid, linkIndex=-1, rgbaColor=[0.2, 0.2, 0.2, 1])

    def _setup_robot(self):
        self._joint_name_to_id = {}
        self._build_joint_name2id_dict()

    def _build_joint_name2id_dict(self):
        num_joints = self._pclient.getNumJoints(self.uid)
        for i in range(num_joints):
            joint_info = self._pclient.getJointInfo(self.uid, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def _reset_joints(self, noisy_state):

        for j_id, name in enumerate(JOINTS_NAMES):
            state_ = STAND_MOTOR_ANGLES[j_id]
            velocity = 0
            if noisy_state:
                # if joint in [0, 3, 6, 9]:
                #     # shoulder joints, should move less
                state_ += np.random.uniform(-np.pi / 16, np.pi / 16)
                velocity += np.random.uniform(-0.05, 0.05)
                # else:
                #     state_ += np.random.uniform(-np.pi/8, np.pi/8)
                #     velocity += np.random.uniform(-0.1, 0.1)

            self._pclient.resetJointState(self.uid, self._joint_name_to_id[name], state_, velocity)

    def action(self, in_action: np.array):
        action = np.array(in_action, dtype=np.float32)
        action = scale(action, 1, -1, UPPER_JOINT_LIMITS, LOWER_JOINT_LIMITS)
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
        positions = scale(positions, UPPER_JOINT_LIMITS, LOWER_JOINT_LIMITS, 1, -1)
        velocities = np.array([j_state[1] for j_state in joint_states_tuple])

        return np.concatenate((positions, velocities))

    def get_action_dim(self):
        return 12

    def get_joint_state_dim(self):
        return 24
