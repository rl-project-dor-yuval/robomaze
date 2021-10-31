import sys
import torch.cuda

sys.path.append('..')
from stable_baselines3.common.env_checker import check_env
import math
from Utils import get_vanilla_navigator_env


nav_env = get_vanilla_navigator_env()

actions = [(2, -math.pi/4)] * 3

obs = nav_env.reset()
for i in range(3):
    obs, _, _, _ = nav_env.step(actions[i])

