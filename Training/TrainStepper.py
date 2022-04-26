import time
import numpy as np
import os
import sys
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

sys.path.append('..')
from MazeEnv.MazeEnv import Rewards
from Utils import make_circular_map, clear_files, get_multi_targets_circle_envs
from Evaluation import EvalAndSaveCallback, MultiTargetEvalAndSaveCallback
import Evaluation
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("running on:", device)