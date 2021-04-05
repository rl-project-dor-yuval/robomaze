import os
import pybullet as p
import time
import pybullet_data
import gym
from gym import error, spaces, utils
from gym.utils import seeding

p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])
p.setGravity(0, 0, -10)
planeId = p.loadURDF("samurai.urdf", basePosition=[0, 0, 0])
cheetahId = p.loadMJCF("mjcf/ant.xml")

#tableId = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])
#trayId = p.loadURDF("tray/traybox.urdf")
#rndId = p.loadURDF("random_urdfs/001/001.urdf")
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])

# Controllers :
#p.setJointMotorControl2(cheetahId, 0, p.POSITION_CONTROL, 0.4)
#p.setJointMotorControl2(cheetahId, 1, p.POSITION_CONTROL, 0.8)
#p.setJointMotorControl2(cheetahId, 2, p.POSITION_CONTROL, 0.5)
#.setJointMotorControl2(cheetahId, 7, p.POSITION_CONTROL, 0.5)


# set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(cheetahId)
print(cubePos, cubeOrn)
p.disconnect()
