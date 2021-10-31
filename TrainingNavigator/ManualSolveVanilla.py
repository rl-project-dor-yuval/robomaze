import sys
import torch.cuda

sys.path.append('..')
from stable_baselines3.common.env_checker import check_env
import math
from Utils import get_vanilla_navigator_env


nav_env = get_vanilla_navigator_env()
nav_env.visualize_mode(True)

actions = [(3, -math.pi/6)] * 5

obs = nav_env.reset()
for i in range(4):
    obs, reward, _, _ = nav_env.step(actions[i])
    print("reward:", reward)

