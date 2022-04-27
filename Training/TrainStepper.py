import time
import numpy as np
import os
import sys
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import wandb

sys.path.append('.')
from MazeEnv.EnvAttributes import Rewards
from Utils import get_multi_targets_circle_envs
from Evaluation import EvalAndSaveCallback, MultiTargetEvalAndSaveCallback
import torch


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("running on:", device)

    # Parameters
    config = {
        "run_name": "Stepper",
        "show_gui": False,
        "seed": 42 ** 3,

        "train_steps": 5_000_000,
        "buffer_size": 300_000,
        "learning_starts": 10_000,
        "timeout_steps": 200,
        "map_radius": 4,
        "learning_rate": 1e-6,
        "reduce_lr": True,
        "lr_reduce_factor": 0.2,
        "exploration_noise_std": 0.05,
        "batch_size": 1024,
        "rewards": Rewards(target_arrival=1, collision=-1, timeout=0, fall=-1, idle=-1e-4),
        "max_goal_velocity": 0.5,
        "target_epsilon": 0.3,

        "eval_freq": 10 ** 5,
        "video_freq": 2

    }
    config["dir"] = "./Training/logs/StepperV2" + config["run_name"]

    # setup W&B:
    wb_run = wandb.init(project="Robomaze-tests", name=config["run_name"],
                        config=config)
    wandb.tensorboard.patch(root_logdir="TrainingNavigator/logs/tb", pytorch=True)

    targets = np.genfromtxt("Training/TestTargets/test_coords_0_7to3_5.csv", delimiter=',')

    maze_env, eval_maze_env = get_multi_targets_circle_envs(radius=config["map_radius"],
                                                            targets=targets,
                                                            timeout_steps=config["timeout_steps"],
                                                            rewards=config["rewards"],
                                                            max_goal_velocity=config["max_goal_velocity"],
                                                            xy_in_obs=False,
                                                            show_gui=config["show_gui"],
                                                            hit_target_epsilon=config["target_epsilon"])

    callback = MultiTargetEvalAndSaveCallback(log_dir=config["dir"],
                                              eval_env=eval_maze_env,
                                              eval_freq=config["eval_freq"],
                                              eval_video_freq=config["video_freq"],
                                              wb_run=wb_run,
                                              verbose=1)

    # create_model
    exploration_noise = NormalActionNoise(mean=np.array([0]*8), sigma=np.array([config["exploration_noise_std"]]*8))

    def lr_func(progress):
        if progress < 0.66 and config["reduce_lr"]:
            return config["learning_rate"] * config["lr_reduce_factor"]
        else:
            return config["learning_rate"]

    model = DDPG(policy="MlpPolicy",
                 env=maze_env,
                 buffer_size=config["buffer_size"],
                 learning_rate=lr_func,
                 batch_size=config["batch_size"],
                 action_noise=exploration_noise,
                 device=device,
                 train_freq=(100, "step"),
                 verbose=0,
                 tensorboard_log="./Training/logs/StepperV2/tb",
                 learning_starts=config["learning_starts"],
                 seed=config["seed"],)

    model.learn(total_timesteps=config["train_steps"], tb_log_name=config["run_name"], callback=callback)

    wb_run.finish()

