import pybullet
from pybullet_utils import bullet_client as bc
import numpy as np
import math

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
INIT_JOINT_STATES = [0, (ANKLE_1_4_HIGH+ANKLE_1_4_LOW)/2,
                     0, (ANKLE_2_3_HIGH+ANKLE_2_3_LOW)/2,
                     0, (ANKLE_2_3_HIGH+ANKLE_2_3_LOW)/2,
                     0, (ANKLE_1_4_HIGH+ANKLE_1_4_LOW)/2]


# this function scales the given value to
def scale(value, old_high, old_low, new_high, new_low):
    return ((value - old_low) / (old_high - old_low)) * (new_high - new_low) + new_low

# TODO: Refactor this position/torque control shit or remove one of them completely

class Ant:
    def __init__(self, pybullet_client, position3d):
        self.start_position = position3d
        self._pclient = pybullet_client
        self.position_control = False

        # load ant and save it's initial orientation,
        # for now this will be the initial orientation always
        self.uid = self._pclient.loadMJCF("ant.xml")[0]

        # color ant:
        for link in range(self._pclient.getNumJoints(self.uid)):
            self._pclient.changeVisualShape(self.uid, linkIndex=link, rgbaColor=[0.4, 0.4, 0.4, 1])
        self._pclient.changeVisualShape(self.uid, linkIndex=-1, rgbaColor=[0.2, 0.2, 0.2, 1])

        self.initial_orientation = self._pclient.getBasePositionAndOrientation(self.uid)[1]

        self.reset()

        # turn off velocity motors, we use torque control
        for joint in JOINTS_INDICES:
            self._pclient.setJointMotorControl2(self.uid, joint, self._pclient.VELOCITY_CONTROL, force=0)

    def reset(self, noisy_state=False):

        initial_orientation = self.initial_orientation
        if noisy_state:
            initial_orientation = self._pclient.getEulerFromQuaternion(initial_orientation)
            initial_orientation = [np.random.uniform(-0.3, 0.3) + oriant for oriant in initial_orientation]
            initial_orientation = self._pclient.getQuaternionFromEuler(initial_orientation)
        self._pclient.resetBasePositionAndOrientation(self.uid,
                                                      self.start_position,
                                                      initial_orientation)
        # start with noisy velocity only if required
        if noisy_state:
            linear_vel = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0]
            angular_vel = np.random.uniform(-0.3, 0.3, 3)
            self._pclient.resetBaseVelocity(self.uid, linear_vel, angular_vel)

        for joint, state in zip(JOINTS_INDICES, INIT_JOINT_STATES):
            state_ = state
            velocity = 0
            if noisy_state:
                state_ += np.random.uniform(-1, 1)
                velocity += np.random.uniform(-0.5, 0.5)
            self._pclient.resetJointState(self.uid, joint, state_, velocity)

    def action(self, in_action: np.array):
        action = np.array(in_action, dtype=np.float32)

        if not self.position_control:  # torque control
            mode = self._pclient.TORQUE_CONTROL
            self._pclient.setJointMotorControlArray(self.uid, JOINTS_INDICES, mode, forces=action*1500)

        else:
            # this is old code from when we used position control and we had to scale actions:

            # action will preform the given action following R8 vector that corresponds to each joint of the ant.
            # will consist as following positions for the relevant joints:
            # [ shoulder1, ankle1,  shoulder2, ankle2, shoulder3, ankle3, shoulder4, ankle4 ]
            # the given position values of the joints are in [-1,1] whereas -1 is the most bended and 1 wide open.
            # in the shoulders joints -1 is rightmost and 1 is left.
            # Practical joints ranges:
            # all Shoulders - [ -0.698 , 0.698 ]
            # Ankle 1,4 (Opposite) - open [ 0.523 , 1.745 ] bend
            # Ankle 2,3 (Opposite) - bend [ -1.745, -0.523 ] open
            #
            #  2  \ /  1
            #      O
            #  3  / \  4


            # scale the given values from the input range to the practical range
            # scale the shoulder position values in the odd indices
            action[::2] = scale(action[::2], 1, -1, SHOULDER_HIGH, SHOULDER_LOW)
            # handle ankles Position
            action[1] = scale(-1 * action[1], 1, -1, ANKLE_1_4_HIGH, ANKLE_1_4_LOW)
            action[7] = scale(-1 * action[7], 1, -1, ANKLE_1_4_HIGH, ANKLE_1_4_LOW)
            action[5] = scale(action[5], 1, -1, ANKLE_2_3_HIGH, ANKLE_2_3_LOW)
            action[3] = scale(action[3], 1, -1, ANKLE_2_3_HIGH, ANKLE_2_3_LOW)

            self._pclient.setJointMotorControlArray(self.uid, JOINTS_INDICES, self._pclient.POSITION_CONTROL,
                                                    action, forces=[2000]*8)

    def get_pos_orientation_velocity(self):
        """
        :return: 12d vector

        3 first values are ant COM position
        3 next values are ant COM velocity
        3 next values are ant euler orientation [Roll, Pitch, Yaw]
        last 3 values are angular velocity
        """
        position, orientation_quat = self._pclient.getBasePositionAndOrientation(self.uid)
        orientation = self._pclient.getEulerFromQuaternion(orientation_quat)
        vel, angular_vel = self._pclient.getBaseVelocity(self.uid)

        return np.concatenate([position, vel, orientation, angular_vel])

    def get_joint_state(self):
        """
        :return: 16d vector

        8 first values are joints position
        8 next values are joint velocity
        """
        joint_states_tuple = self._pclient.getJointStates(self.uid, JOINTS_INDICES)
        positions = np.array([j_state[0] for j_state in joint_states_tuple])
        velocities = np.array([j_state[1] for j_state in joint_states_tuple])

        positions[::2] = scale(positions[::2], SHOULDER_HIGH, SHOULDER_LOW, 1, -1,)
        positions[1] = -scale(positions[1], ANKLE_1_4_HIGH, ANKLE_1_4_LOW, 1, -1)
        positions[7] = -scale(positions[7], ANKLE_1_4_HIGH, ANKLE_1_4_LOW, 1, -1)
        positions[5] = scale(positions[5], ANKLE_2_3_HIGH, ANKLE_2_3_LOW, 1, -1)
        positions[3] = scale(positions[3], ANKLE_2_3_HIGH, ANKLE_2_3_LOW, 1, -1)

        return np.concatenate((positions, velocities))

    def set_position_control(self, position_control):
        self.position_control = position_control

        if not self.position_control:
            # turn off velocity motors, we use torque control
            for joint in JOINTS_INDICES:
                self._pclient.setJointMotorControl2(self.uid, joint, self._pclient.VELOCITY_CONTROL, force=0)


