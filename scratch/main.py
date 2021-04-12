import os
import pybullet as p
import time
import pybullet_data
import gym
from gym import error, spaces, utils
from gym.utils import seeding

p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])
p.setGravity(0, 0, -10)
planeId = p.loadURDF("samurai.urdf", basePosition=[0, 0, 0])
spiderId = p.loadMJCF("mjcf/ant.xml")[0]

for i in range(-7,8):
    cubeId = p.loadURDF("cube.urdf", basePosition=[i, 7, 0.5])
    cubeId = p.loadURDF("cube.urdf", basePosition=[i, -7, 0.5])
    cubeId = p.loadURDF("cube.urdf", basePosition=[-7, i, 0.5])
    cubeId = p.loadURDF("cube.urdf", basePosition=[7, i, 0.5])


#trayId = p.loadURDF("tray/traybox.urdf")
#rndId = p.loadURDF("random_urdfs/001/001.urdf")
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])

#There are different types of Joints:
#JOINT_REVOLUTE - 0 - כמו מרפק של יד
#JOINT_PRISMATIC - 1
#JOINT_SPHERICAL - 2
#JOINT_PLANER - 3
#JOINT_FIXED - 4
# In the spider all of the JOINTS ARE REVOLUTE AND FIXED
# How to recieve info about Joints:
NumJoints = p.getNumJoints(spiderId)
for j in range(NumJoints):
    print(p.getJointInfo(spiderId, j), "\n")


#Every Arm has 5 joints , arms are devided from 1 to 4:
# Arm 1 : joint 0 - 4
# Arm 2 : joint 5 - 9
# Arm 3 : joint 10-14
# Arm 4 : joint 15-19
Arm_1 = [i for i in range(5)]
Arm_2 = [i for i in range(5,10)]
Arm_3 = [i for i in range(10,15)]
Arm_4 = [i for i in range(15,20)]

# mode = p.VELOCITY_CONTROL
mode = p.POSITION_CONTROL

# Controllers :      # BodyID ,  joint idx, controlMode, targetPosition (optional)
# Control each individual joint
p.setJointMotorControl2(spiderId, 5, mode, 0.9,force = 1000)
p.setJointMotorControl2(spiderId, 6, mode, 0.8, force =1000)
p.setJointMotorControl2(spiderId, 7, mode, 0.9, force = 1000)
p.setJointMotorControl2(spiderId, 8, mode, 0.8, force = 1000)
p.setJointMotorControl2(spiderId, 9, mode, -1, force = 1000)

# Control array of joint indices
lst = [1, 1, 1, 1, 1]
Positions = [float(i) for i in lst]

# Controllers :            #BodyID    joint indices cotrolMode Positions float(arr)
p.setJointMotorControlArray(spiderId, Arm_1, mode, targetPositions=Positions)
p.setJointMotorControlArray(spiderId, Arm_4, mode, targetPositions=Positions)



# Get joint states
print("Joints 5-9 states")
for i in range(5,10):
    print(p.getJointState(spiderId, i), "\n")

print("num Bodies:",p.getNumBodies())
# Tuning Parameters to change Spider Position
shoulder1 = p.addUserDebugParameter("shoulder1", -1, 1, 0)
elbow1 = p.addUserDebugParameter("elbow1", -1, 1, 0)

shoulder2 = p.addUserDebugParameter("shoulder2", -1, 1, 0)
elbow2 = p.addUserDebugParameter("elbow2", -1, 1, 0)

shoulder3 = p.addUserDebugParameter("shoulder3", -1, 1, 0)
elbow3 = p.addUserDebugParameter("elbow3", -1, 1, 0)

shoulder4 = p.addUserDebugParameter("shoulder4", -1, 1, 0)
elbow4 = p.addUserDebugParameter("elbow4", -1, 1, 0)


for i in range(10000):
    p.stepSimulation()

    # setting the sliders to the joints
    s1 = p.readUserDebugParameter(shoulder1)
    e1= p.readUserDebugParameter(elbow1)
    s2 = p.readUserDebugParameter(shoulder2)
    e2= p.readUserDebugParameter(elbow2)
    s3 = p.readUserDebugParameter(shoulder3)
    e3= p.readUserDebugParameter(elbow3)
    s4 = p.readUserDebugParameter(shoulder4)
    e4= p.readUserDebugParameter(elbow4)

    #setting the value of the joint to the compatible variable
    p.setJointMotorControl2(spiderId, 1, mode, s1, force=1000)
    p.setJointMotorControl2(spiderId, 3, mode, e1, force=1000)
    p.setJointMotorControl2(spiderId, 6, mode, s2, force=1000)
    p.setJointMotorControl2(spiderId, 8, mode, e2, force=1000)
    p.setJointMotorControl2(spiderId, 11, mode, s3, force=1000)
    p.setJointMotorControl2(spiderId, 13, mode, e3, force=1000)
    p.setJointMotorControl2(spiderId, 16, mode, s4, force=1000)
    p.setJointMotorControl2(spiderId, 18, mode, e4, force=1000)


    time.sleep(1. / 240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(spiderId)
print(cubePos, cubeOrn)
p.disconnect()
