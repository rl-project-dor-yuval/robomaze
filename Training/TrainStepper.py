import time
import numpy as np
import os
import sys
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append('..')
from MazeEnv.MazeEnv import Rewards
from Utils import get_multi_targets_circle_envs
from Evaluation import EvalAndSaveCallback, MultiTargetEvalAndSaveCallback
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("running on:", device)

# Parameters
config = {
    "run_name": "Stepper_first_try",
    "show_gui": True,
    "seed": 42 ** 3,

    "train_steps": 300_000,
    "timeout_steps": 200,
    "map_radius": 4,

    "learning_rate": 1e-6,
    "reduce_lr": True,
    "exploration_noise_std": 0.4,
    "batch_size": 1024,
    "rewards": Rewards(target_arrival=1, collision=-1, timeout=0, fall=-1, idle=-0.2e-3),
    "eval_freq": 10 ** 4,
    "video_freq": 25

}
config["dir"] = "./Training/StepperV2/logs/" + config["run_name"]

# setup W&B:
# wb_run = wandb.init(project="Robomaze-Stepper-tests", name=config["run_name"],
#                     config=config)
# wandb.tensorboard.patch(root_logdir="TrainingNavigator/logs/tb", pytorch=True)

targets = np.genfromtxt("TestTargets/test_coords_0_7to3_5.csv", delimiter=',')

maze_env, eval_maze_env = get_multi_targets_circle_envs(radius=config["map_radius"],
                                                        targets=targets,
                                                        timeout_steps=config["timeout_steps"],
                                                        rewards=config["rewards"],
                                                        xy_in_obs=False,
                                                        show_gui=config["show_gui"])

callback = MultiTargetEvalAndSaveCallback(log_dir=config["dir"],
                                          eval_env=eval_maze_env,
                                          eval_freq=config["eval_freq"],
                                          eval_video_freq=config["video_freq"],
                                          verbose=1)

# create_model
# exploration_noise = NormalActionNoise(mean=np.array([0]*8), sigma=np.array([config["exploration_noise_std"]]*8))
# policy_kwargs = dict(net_arch=dict(pi=config["actor_arch"], qf=config["critic_arch"]))
#
# model = DDPG(policy="MlpPolicy",
#              env=maze_env,
#              buffer_size=config["buffer_size"],
#              learning_rate=config["learning_rate"],
#              batch_size=config["batch_size"],
#              action_noise=exploration_noise,
#              device=device,
#              train_freq=(1, "episode"),
#              verbose=0,
#              tensorboard_log="./StepperV2/logs/tb",
#              learning_starts=config["learning_starts"],
#              seed=config["seed"],
#              policy_kwargs=policy_kwargs)
#
# model.learn(total_timesteps=config["train_steps"], tb_log_name=config["run_name"], callback=callback)
#
# wb_run.finish()

if __name__ == "__main__":

    model = torch.load(".\TrainingNavigator\StepperAgent.pt")
    maze_env.reset(target_index=1, create_video=False)
    is_done = False
    while is_done is False:
        action = model.predict(maze_env.get_observation())
        obs, reward, is_done, _ = maze_env.step(action)

        if reward != 0:
            print(reward)
        time.sleep(1. / 20)
