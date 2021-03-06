"""
Usage: TrainingMultiproc.py ConfigFileName.yaml
the config file must appear in TrainingNavigator/Configs and you shouldn't pass the path
"""

import sys
sys.path.append('.')
from TrainingNavigator.TD3MP import TD3MP, CustomTD3Policy
import argparse
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from TrainingNavigator.NavigatorEnv import MultiStartgoalNavigatorEnv
from MazeEnv.MazeEnv import MazeEnv
from DDPGMP import DDPGMP
import torch
from MazeEnv.EnvAttributes import Rewards
import wandb
from TrainingNavigator.NavEvaluation import NavEvalCallback
import yaml
from Utils import blackwhiteswitch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', type=str, help='config file name without path,'
                                                      ' the file must appear in TrainingNavigator/configs/')
    args = parser.parse_args()

    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("TrainingNavigator/configs/" + args.config_name, "r"), yaml_loader)
    print(config)
    # noinspection DuplicatedCode
    config["dir"] = "./TrainingNavigator/logs/" + config["run_name"]
    # ---

    # setup W&B:
    wb_run = wandb.init(project=config["project"], name=config["run_name"],
                        group=config["group"], config=config)
    wandb.tensorboard.patch(root_logdir="TrainingNavigator/logs/tb", pytorch=True)

    # Setup Training Environment
    maze_map = blackwhiteswitch(config["maze_map_path"])
    start_goal_pairs = np.load(config["workspaces_path"]) / 10  # all maps granularity is 10
    validation_start_goal_pairs = np.load(config["validation_workspaces_path"]) / 10

    maze_env_kwargs = dict(maze_size=config["maze_size"], maze_map=maze_map, start_loc=start_goal_pairs[0][0],
                           target_loc=start_goal_pairs[0][-1], xy_in_obs=True,
                           show_gui=config["show_gui"], )
    nav_env_kwargs = dict(start_goal_pairs=start_goal_pairs,
                          maze_env_kwargs=maze_env_kwargs,
                          epsilon_to_hit_subgoal=config["epsilon_to_subgoal"],
                          epsilon_rotation_to_hit_subgoal=config["epsilon_rotation_to_subgoal"],
                          max_vel_in_subgoal=config["max_velocity_in_subgoal"],
                          rewards=config["rewards"],
                          done_on_collision=config["done_on_collision"],
                          max_stepper_steps=config["max_stepper_steps"],
                          max_steps=config["max_navigator_steps"],
                          stepper_radius_range=config["stepper_radius_range"],
                          stepper_agent=config["stepper_agent_path"],
                          wall_hit_limit=config["wall_hit_limit"],
                          repeat_failed_ws_prob=config["repeat_failed_ws_prob"],)

    nav_env = make_vec_env(MultiStartgoalNavigatorEnv, n_envs=config["num_envs"], seed=config["seed"],
                           env_kwargs=nav_env_kwargs, vec_env_cls=SubprocVecEnv)
    nav_env.env_method('visualize_mode', False)

    # noinspection DuplicatedCode
    # set up separate evaluation environment:
    raise NotImplementedError("Fix to new workspaces")
    eval_maze_env = MazeEnv(maze_size=config["maze_size"], maze_map=maze_map, start_loc=start_goal_pairs[0][0],
                            target_loc=validation_start_goal_pairs[0][-1], xy_in_obs=True, show_gui=False)
    eval_nav_env = MultiStartgoalNavigatorEnv(start_goal_pairs=validation_start_goal_pairs,
                                              maze_env=eval_maze_env,
                                              epsilon_to_hit_subgoal=config["epsilon_to_subgoal"],
                                              max_vel_in_subgoal=config["max_velocity_in_subgoal"],
                                              rewards=config["rewards"],
                                              done_on_collision=config["done_on_collision"],
                                              max_stepper_steps=config["max_stepper_steps"],
                                              max_steps=config["max_navigator_steps"],
                                              stepper_agent=config["stepper_agent_path"],
                                              stepper_radius_range=config["stepper_radius_range"],
                                              wall_hit_limit=config["wall_hit_limit"])
    # noinspection DuplicatedCode
    eval_nav_env.visualize_mode(False)

    # set up model and run:
    exploration_noise = NormalActionNoise(mean=np.array([0] * 3),
                                          sigma=np.array([config["exploration_noise_std"]] * 3))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('running on:', device)

    policy_kwargs = dict(net_arch=dict(pi=config["actor_arch"], qf=config["critic_arch"]))
    model_kwargs = dict(policy=CustomTD3Policy,
                        env=nav_env,
                        buffer_size=config["buffer_size"],
                        learning_rate=config["learning_rate"],
                        batch_size=config["batch_size"],
                        action_noise=exploration_noise,
                        device=device,
                        train_freq=(500, "step"),
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
    if config["use_TD3MP"]:
        model = TD3MP(**model_kwargs)
    else:
        model = DDPGMP(**model_kwargs)

    eval_freq2 = config["eval_freq2"] if "eval_freq2" in config else -1
    change_eval_freq_after = config["change_eval_freq_after"] if "change_eval_freq_after" in config else -1
    callback = NavEvalCallback(dir=config["dir"],
                               eval_env=eval_nav_env,
                               wandb_run=wb_run,
                               validation_traj_path=config["validation_demonstration_path"],
                               eval_freq=config["eval_freq"],
                               eval_video_freq=config["video_freq"],
                               save_model_freq=config["save_model_freq"],
                               eval_workspaces=config["eval_workspaces"],
                               maze_map=maze_map,
                               eval_freq2=eval_freq2,
                               change_eval_freq_after=change_eval_freq_after,
                               verbose=1)

    model.learn(total_timesteps=config["train_steps"], tb_log_name=config["run_name"], callback=callback)

    wb_run.finish()
