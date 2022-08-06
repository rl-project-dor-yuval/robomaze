from abc import abstractmethod
from pybullet_utils import bullet_client as bc
import numpy as np


POINTER_COLOR = [0, 1, 0, 0.5]


# this function scales the given value to new range
def scale(value, old_high, old_low, new_high, new_low):
    return ((value - old_low) / (old_high - old_low)) * (new_high - new_low) + new_low


class RobotBase:
    """ a base class for all robots. a robot must implement all the abstract method at the bottom of this class """

    def __init__(self, pybullet_client: bc, position3d: np.ndarray, heading: float, robot_filename: str,
                 scale_urdf: float):
        self._pclient = pybullet_client
        self.start_position = position3d
        self.heading = heading
        self.position_control = False

        if robot_filename.endswith('.xml'):
            self.uid = self._pclient.loadMJCF(robot_filename)[0]
        else:
            self.uid = self._pclient.loadURDF(robot_filename, globalScaling=scale_urdf)

        initial_orientation = self._pclient.getBasePositionAndOrientation(self.uid)[1]
        initial_orientation_euler = list(self._pclient.getEulerFromQuaternion(initial_orientation))
        initial_orientation_euler[2] = heading
        self.initial_orientation = self._pclient.getQuaternionFromEuler(initial_orientation_euler)

        pointer_orientation = self._pclient.getQuaternionFromEuler((0, 0, heading))
        pointer_position = list(position3d)
        pointer_position[2] += 0.75
        self._direction_pointer = self._pclient.loadURDF("direction_pointer.urdf",
                                                         basePosition=pointer_position,
                                                         baseOrientation=pointer_orientation,
                                                         globalScaling=1.25)
        self._pclient.setCollisionFilterGroupMask(self._direction_pointer, -1, 0, 0)  # disable collisions
        self._pclient.changeVisualShape(self._direction_pointer, -1, rgbaColor=[0, 0, 0, 1])

        self.reset()

        # turn off velocity motors, we use torque control
        for joint in range(self._pclient.getNumJoints(self.uid)):
            self._pclient.setJointMotorControl2(self.uid, joint, self._pclient.VELOCITY_CONTROL, force=0)

    def reset(self, noisy_state=False):

        initial_orientation = self.initial_orientation
        if noisy_state:
            initial_orientation = list(self._pclient.getEulerFromQuaternion(initial_orientation))
            initial_orientation = [np.random.uniform(-0.3, 0.3) + oriant for oriant in initial_orientation[:2]] + \
                                  initial_orientation[2:]
            initial_orientation = self._pclient.getQuaternionFromEuler(initial_orientation)
        self._pclient.resetBasePositionAndOrientation(self.uid,
                                                      self.start_position,
                                                      initial_orientation)
        # start with noisy velocity only if required
        if noisy_state:
            linear_vel = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0]
            angular_vel = np.random.uniform(-0.3, 0.3, 3)
            self._pclient.resetBaseVelocity(self.uid, linear_vel, angular_vel)

        self._reset_joints(noisy_state)

        self.update_direction_pointer()

    def update_direction_pointer(self, visible=True):
        if visible:
            position, orientation_quat = self._pclient.getBasePositionAndOrientation(self.uid)
            orientation = self._pclient.getEulerFromQuaternion(orientation_quat)

            pointer_position = list(position)
            pointer_position[2] += 0.75

            self._pclient.resetBasePositionAndOrientation(self._direction_pointer, pointer_position,
                                                          self._pclient.getQuaternionFromEuler((0, 0, orientation[2])))
            self._pclient.changeVisualShape(self._direction_pointer, -1, rgbaColor=POINTER_COLOR)
        else:
            self._pclient.changeVisualShape(self._direction_pointer, -1, rgbaColor=[0, 0, 0, 0])

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

    def set_start_state(self, position3d, heading):
        self.start_position = position3d
        orientation_euler = list(self._pclient.getEulerFromQuaternion(self.initial_orientation))
        orientation_euler[2] = heading
        self.initial_orientation = self._pclient.getQuaternionFromEuler(orientation_euler)

    def get_state_dim(self):
        return 15 + self.get_joint_state_dim()
        # 15 is the number of dimensions of the position and orientation and velocity
        # and relative angle to target, relative distance and angle difference to desired

    @abstractmethod
    def _reset_joints(self, noisy_state):
        raise NotImplementedError("any robot must implement _reset_joints")

    @abstractmethod
    def action(self, in_action: np.array):
        raise NotImplementedError("any robot must implement action")

    @abstractmethod
    def get_joint_state(self):
        """
        must return a vector of joint states followed by joint velocities
        """
        raise NotImplementedError("any robot must implement get_joint_state")

    @abstractmethod
    def get_action_dim(self):
        raise NotImplementedError("any robot must implement get_action_dim")

    @abstractmethod
    def get_joint_state_dim(self):
        raise NotImplementedError("any robot must implement _get_joint_state_dim")


