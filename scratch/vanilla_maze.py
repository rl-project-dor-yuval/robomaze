import sys
import torch.cuda

sys.path.append('..')
import cv2
import MazeEnv.MazeEnv as mz
import time
from TrainingNavigator.StepperAgent import StepperAgent
from TrainingNavigator.NavigatorEnv import NavigatorEnv
from stable_baselines3.common.env_checker import check_env
from gym.wrappers.rescale_action import RescaleAction

maze_map = - (cv2.imread("vanilla_map.png", cv2.IMREAD_GRAYSCALE) / 255) + 1
maze_map = maze_map.T

env = mz.MazeEnv(maze_size=mz.MazeSize.SQUARE10,
                 maze_map=maze_map,
                 tile_size=0.05,
                 start_loc=(1., 7.5),
                 target_loc=(4.8, 4.2),
                 xy_in_obs=True,
                 show_gui=True)  # missing, timeout, rewards

device = 'auto'
# naively try to solve it:
agent = StepperAgent("../TrainingNavigator/StepperAgent.pt", device=device)

nav_env = NavigatorEnv(maze_env=env,
                      stepper_agent=agent, )

wrapped_nav_env = RescaleAction(nav_env, -1, 1)
check_env(wrapped_nav_env)

obs = nav_env.reset()

