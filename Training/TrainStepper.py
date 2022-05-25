"""
Usage: TrainStepper.py ConfigFileName.yaml
the config file must appear in Training/Configs and you shouldn't pass the full path
"""

import argparse
import time
import numpy as np
import os
import sys
import yaml
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
import wandb

sys.path.append('.')
from MazeEnv.EnvAttributes import Rewards
from Utils import get_multi_targets_circle_envs, get_multi_targets_circle_envs_multiproc
from Evaluation import EvalAndSaveCallback, MultiTargetEvalAndSaveCallback
import torch

if __name__ == '__main__':
    # noinspection DuplicatedCode
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', type=str, help='config file name without path,'
                                                      ' the file must appear in Training/configs/')
    args = parser.parse_args()

    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("Training/configs/" + args.config_name, "r"), yaml_loader)
    config["dir"] = "./Training/logs/StepperV2" + config["run_name"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("running on:", device)

    # setup W&B:
    wb_run = wandb.init(project="Robomaze-TrainingStepper", name=config["run_name"],
                        config=config)
    wandb.tensorboard.patch(root_logdir="Training/logs/StepperV2/tb", pytorch=True)

    targets = np.genfromtxt("Training/TestTargets/test_coords_0_6to3.5.csv", delimiter=',')

    env_kwargs = dict(radius=config["map_radius"],
                      targets=targets,
                      timeout_steps=config["timeout_steps"],
                      rewards=config["rewards"],
                      max_goal_velocity=config["max_goal_velocity"],
                      xy_in_obs=False,
                      show_gui=config["show_gui"],
                      hit_target_epsilon=config["target_epsilon"],
                      random_ant_initialization=config["random_initialization"],
                      with_obstacles=config["with_obstacles"])
    if config["num_envs"] == 1:
        maze_env, eval_maze_env = get_multi_targets_circle_envs(**env_kwargs)
    else:
        print("training stepper with multiple envrionments is inefficient."
              " if you want to do this anyway, remove this line and make sure "
              "get_multi_targets_circle_envs_multiproc is updated")
        exit(0)
        maze_env, eval_maze_env = get_multi_targets_circle_envs_multiproc(**env_kwargs,
                                                                          num_envs=config["num_envs"])

    callback = MultiTargetEvalAndSaveCallback(log_dir=config["dir"],
                                              eval_env=eval_maze_env,
                                              eval_freq=config["eval_freq"],
                                              eval_video_freq=config["video_freq"],
                                              wb_run=wb_run,
                                              verbose=1)

    # create_model
    noise_sigma = [config["exploration_noise_std_shoulder"], config["exploration_noise_std_ankle"]] * 4
    exploration_noise = NormalActionNoise(mean=np.array([0] * 8), sigma=np.array(noise_sigma))


    def lr_func(progress):
        if progress < 0.33 and config["reduce_lr"]:
            return config["learning_rate"] * config["lr_reduce_factor"]
        else:
            return config["learning_rate"]


    model = TD3(policy="MlpPolicy",
                env=maze_env,
                buffer_size=config["buffer_size"],
                learning_rate=lr_func,
                batch_size=config["batch_size"],
                action_noise=exploration_noise,
                device=device,
                train_freq=(1, "episode"),
                verbose=0,
                tensorboard_log="./Training/logs/StepperV2/tb",
                learning_starts=config["learning_starts"],
                gamma=config["gamma"],
                seed=config["seed"],
                policy_delay=2,
                tau=config["tau"],)

    model.learn(total_timesteps=config["train_steps"], tb_log_name=config["run_name"], callback=callback)

    wb_run.finish()
