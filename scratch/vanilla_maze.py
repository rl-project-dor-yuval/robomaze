import sys
sys.path.append('..')
import cv2
import MazeEnv.MazeEnv as mz
import time
from TrainingNavigator.WalkerAgent import WalkerAgent

maze_map = - (cv2.imread("vanilla_map.png", cv2.IMREAD_GRAYSCALE) / 255) + 1
maze_map = maze_map.T

env = mz.MazeEnv(maze_size=mz.MazeSize.SQUARE10,
                 maze_map=maze_map,
                 tile_size=0.05,
                 start_loc=(1, 7.5),
                 target_loc=(3, 5),
                 xy_in_obs=False,
                 show_gui=True)  # missing, timeout, rewards

# naively try to solve it:
agent = WalkerAgent("../TrainingNavigator/WalkerAgent.pt")
obs = env.reset()
for i in range(10000):
    action = agent.step(obs)
    obs, _, _, _ = env.step(action)
    time.sleep(1/24.)
