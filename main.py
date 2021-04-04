import pybullet as p
import time
import pybullet_data

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf", basePosition=[0,0,0])
tableId = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0.1])
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
itemId = p.loadURDF("random_urdfs/000/000.urdf", basePosition=[0,0,2], baseOrientation=startOrientation)
# set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(itemId)
print(cubePos, cubeOrn)
p.disconnect()
