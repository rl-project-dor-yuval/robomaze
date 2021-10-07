import pybullet
from pybullet_utils import bullet_client as bc
import numpy as np
import math

# the indices of the joints in the built model
ANKLE_IDX = np.array([3, 8, 13, 18])
SHOULDER_IDX = np.array([1, 6, 11, 16])
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


class Ant:
    def __init__(self, pybullet_client, position3d):
        self.start_position = position3d

        self._pclient = pybullet_client
        # load ant and save it's initial orientation,
        # for now this will be the initial orientation always
        self.uid = self._pclient.loadMJCF("ant.xml")[0]
        self.initial_orientation = self._pclient.getBasePositionAndOrientation(self.uid)[1]
        self.reset()

        # Initializing ant's action space, for 8 joint ranging -1 to 1.

    def reset(self):
        self._pclient.resetBasePositionAndOrientation(self.uid,
                                          self.start_position,
                                          self.initial_orientation)
        for joint, state in zip(JOINTS_INDICES, INIT_JOINT_STATES):
            self._pclient.resetJointState(self.uid, joint, state)

    def action(self, in_action: np.array):
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
        # assert in_action.dtype == 'float64', "action dtype is not float64"

        mode = self._pclient.POSITION_CONTROL
        action = np.copy(in_action)
        # scale the given values from the input range to the practical range
        # scale the shoulder position values in the odd indices
        action[::2] = scale(action[::2], 1, -1, SHOULDER_HIGH, SHOULDER_LOW)

        # handle ankles Position
        action[1] = scale(-1 * action[1], 1, -1, ANKLE_1_4_HIGH, ANKLE_1_4_LOW)
        action[7] = scale(-1 * action[7], 1, -1, ANKLE_1_4_HIGH, ANKLE_1_4_LOW)
        action[5] = scale(action[5], 1, -1, ANKLE_2_3_HIGH, ANKLE_2_3_LOW)
        action[3] = scale(action[3], 1, -1, ANKLE_2_3_HIGH, ANKLE_2_3_LOW)

        # perform the move
        self._pclient.setJointMotorControlArray(self.uid, JOINTS_INDICES, mode, action, forces=[2000]*8)

    def get_pos_vel_and_facing_direction(self):
        """
        return 5d vector,
        2 first values are ant position
        2 next values are ant velocity
        last value the facing direction of the ant
        """
        position, orientation_quat = self._pclient.getBasePositionAndOrientation(self.uid)
        orientation = self._pclient.getEulerFromQuaternion(orientation_quat)
        vel, _ = self._pclient.getBaseVelocity(self.uid)
        # we only take x and y velocity and position
        # yaw (rotation around z) is in index 2 of the orientation
        return np.array([position[0], position[1], vel[0], vel[1], orientation[2]])

    def get_joint_state(self):
        """
         return 16d vector,
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

