import sys

sys.path.append('.')

import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from TrainingNavigator.NavigatorEnv import MultiStartgoalNavigatorEnv
from MazeEnv.MazeEnv import MazeEnv
import cv2
from DDPGMP import DDPGMP, CustomTD3Policy
import torch
from MazeEnv.EnvAttributes import Rewards
import wandb
from TrainingNavigator.NavEvaluation import NavEvalCallback
from TrainingNavigator.StepperAgent import StepperAgent

if __name__ == '__main__':
    # --- Parameters
    config = {
        "run_name": "EvenLessClip_eps0.3_vel0.5_RandInitStepperdifSeed",
        "project": "Robomaze-TrainingNavigator",  # "Robomaze-tests"
        "show_gui": False,
        "seed": 42,
        "train_steps": 5 * 10 ** 6,

        # Training and environment parameters
        "num_envs": 3,
        "learning_rate": 0.5e-5,
        "grad_clip_norm_actor": 40,
        "grad_clip_norm_critic": 1.5,
        "batch_size": 2048,
        "buffer_size": 1 * 10 ** 5,
        "actor_arch": [400, 300],  # Should not be changed or explored
        "critic_arch": [400, 300],  # Should not be changed or explored
        "exploration_noise_std": 0.03,
        "epsilon_to_subgoal": 0.3,  # DO NOT TOUCH
        "max_velocity_in_subgoal": 0.5,  # DO NOT TOUCH
        "stepper_radius_range": (0.4, 2.5),  # DO NOT TOUCH
        "done_on_collision": True,  # modify rewards in case you change this
        "rewards": Rewards(target_arrival=1, collision=-1, fall=-1, idle=-0.001, ),
        "demonstration_path": 'TrainingNavigator/workspaces/bottleneckXL_short1.5_trajectories.npz',
        "demo_on_fail_prob": 0.2,
        "demo_prob_decay": 0.999,
        "use_demo_epsilon_offset": False,
        "learning_starts": 10 ** 4,

        "stepper_agent_path": 'TrainingNavigator/StepperAgents/StepperV2_ep03_vel05_randInit.pt',

        "velocity_in_obs": False,
        "max_stepper_steps": 75,
        "max_navigator_steps": 100,

        # logging parameters
        "eval_workspaces": 100,  # will take the first workspaces
        "eval_freq": 10000,
        "video_freq": 1,
        "save_model_freq": 20000,

        # Constants:
        "maze_size": (10, 10)
    }
    # noinspection DuplicatedCode
    config["dir"] = "./TrainingNavigator/logs/" + config["run_name"]
    # ---

    # setup W&B:
    wb_run = wandb.init(project=config["project"], name=config["run_name"],
                        config=config)
    wandb.tensorboard.patch(root_logdir="TrainingNavigator/logs/tb", pytorch=True)

    # Setup Training Environment
    maze_map = - (cv2.imread('TrainingNavigator/maps/bottleneck.png', cv2.IMREAD_GRAYSCALE) / 255) + 1

    start_goal_pairs = np.load('TrainingNavigator/workspaces/bottleneckXL.npy') / config["maze_size"][0]

    maze_env_kwargs = dict(maze_size=config["maze_size"], maze_map=maze_map, start_loc=start_goal_pairs[0][0],
                           target_loc=start_goal_pairs[0][-1], xy_in_obs=True,
                           show_gui=config["show_gui"], )
    nav_env_kwargs = dict(start_goal_pairs=start_goal_pairs,
                          maze_env_kwargs=maze_env_kwargs,
                          epsilon_to_hit_subgoal=config["epsilon_to_subgoal"],
                          max_vel_in_subgoal=config["max_velocity_in_subgoal"],
                          rewards=config["rewards"],
                          done_on_collision=config["done_on_collision"],
                          max_stepper_steps=config["max_stepper_steps"],
                          max_steps=config["max_navigator_steps"],
                          stepper_radius_range=config["stepper_radius_range"],
                          velocity_in_obs=config["velocity_in_obs"],
                          stepper_agent=config["stepper_agent_path"],)

    nav_env = make_vec_env(MultiStartgoalNavigatorEnv, n_envs=config["num_envs"], seed=config["seed"],
                           env_kwargs=nav_env_kwargs, vec_env_cls=SubprocVecEnv)
    nav_env.env_method('visualize_mode', False)

    # noinspection DuplicatedCode
    # set up separate evaluation environment:
    eval_maze_env = MazeEnv(maze_size=config["maze_size"], maze_map=maze_map, start_loc=start_goal_pairs[0][0],
                            target_loc=start_goal_pairs[0][-1], xy_in_obs=True, show_gui=False)
    eval_nav_env = MultiStartgoalNavigatorEnv(start_goal_pairs=start_goal_pairs,
                                              maze_env=eval_maze_env,
                                              epsilon_to_hit_subgoal=config["epsilon_to_subgoal"],
                                              max_vel_in_subgoal=config["max_velocity_in_subgoal"],
                                              rewards=config["rewards"],
                                              done_on_collision=config["done_on_collision"],
                                              max_stepper_steps=config["max_stepper_steps"],
                                              max_steps=config["max_navigator_steps"],
                                              velocity_in_obs=config["velocity_in_obs"],
                                              stepper_agent=config["stepper_agent_path"],
                                              stepper_radius_range=config["stepper_radius_range"])
    # noinspection DuplicatedCode
    eval_nav_env.visualize_mode(False)

    # set up model and run:
    exploration_noise = NormalActionNoise(mean=np.array([0] * 2),
                                          sigma=np.array([config["exploration_noise_std"]] * 2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('running on:', device)

    policy_kwargs = dict(net_arch=dict(pi=config["actor_arch"], qf=config["critic_arch"]))
    model = DDPGMP(policy=CustomTD3Policy,
                   env=nav_env,
                   buffer_size=config["buffer_size"],
                   learning_rate=config["learning_rate"],
                   batch_size=config["batch_size"],
                   action_noise=exploration_noise,
                   device=device,
                   train_freq=(100, "step"),
                   verbose=0,
                   tensorboard_log="./TrainingNavigator/logs/tb",
                   learning_starts=config["learning_starts"],
                   seed=config["seed"],
                   demonstrations_path=config["demonstration_path"],
                   demo_on_fail_prob=config["demo_on_fail_prob"],
                   grad_clip_norm_actor=config["grad_clip_norm_actor"],
                   grad_clip_norm_critic=config["grad_clip_norm_critic"],
                   demo_prob_decay=config["demo_prob_decay"],
                   use_demo_epsilon_offset=config["use_demo_epsilon_offset"],
                   policy_kwargs=policy_kwargs)

    callback = NavEvalCallback(dir=config["dir"],
                               eval_env=eval_nav_env,
                               wandb_run=wb_run,
                               eval_freq=config["eval_freq"],
                               eval_video_freq=config["video_freq"],
                               save_model_freq=config["save_model_freq"],
                               eval_workspaces=config["eval_workspaces"],
                               maze_map=maze_map,
                               verbose=1)

    model.learn(total_timesteps=config["train_steps"], tb_log_name=config["run_name"], callback=callback)

    wb_run.finish()
