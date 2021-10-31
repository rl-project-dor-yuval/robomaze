import sys
import os
import cv2
import MazeEnv.MazeEnv as mz
from TrainingNavigator.StepperAgent import StepperAgent
from TrainingNavigator.NavigatorEnv import NavigatorEnv
from stable_baselines3.common.env_checker import check_env
from gym.wrappers.rescale_action import RescaleAction

def get_vanilla_navigator_env(start_loc=(1., 7.5), target_loc=(9, 3)):
    """
    create a navigator env with the vanilla maze to solve,
    NOT wrapped with RescaleAction
    """
    map_path = os.path.join(os.path.dirname(sys.argv[0]), "vanilla_map.png")
    print(map_path)
    maze_map = - (cv2.imread(map_path, cv2.IMREAD_GRAYSCALE) / 255) + 1
    maze_map = maze_map.T

    env = mz.MazeEnv(maze_size=mz.MazeSize.SQUARE10,
                     maze_map=maze_map,
                     tile_size=0.05,
                     start_loc=start_loc,
                     target_loc=target_loc,
                     xy_in_obs=True,
                     show_gui=True)  # missing, timeout, rewards

    agent_path = os.path.join(os.path.dirname(sys.argv[0]), "StepperAgent.pt")
    agent = StepperAgent(agent_path)

    return NavigatorEnv(maze_env=env, stepper_agent=agent, )


def get_vanilla_navigator_env_scaled(start_loc=(1., 7.5), target_loc=(9, 3)):
    """
    create a navigator env with the vanilla maze to solve,
    action space is scaled to [-1, 1]
    """
    env = get_vanilla_navigator_env()
    return RescaleAction(env, -1, 1)


def test_navigator_envrionment():
    """
    check if the implementation of NavigatorEnv is valid using sb3 check_env()
    """

    check_env(get_vanilla_navigator_env_scaled())

