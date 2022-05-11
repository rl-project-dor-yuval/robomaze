import time
from pathlib import Path
import os

import cv2
import torch
import pandas as pd
import numpy as np
from Utils import blackwhiteswitch

from TrainingNavigator.NavigatorEnv import NavigatorEnv, MultiStartgoalNavigatorEnv
from MazeEnv.MazeEnv import MazeEnv
from MazeEnv.EnvAttributes import Rewards
from DDPGMP import DDPGMP

"""
This script is used to track statistics of the trained navigator.
it shows the model's reasons for failure and success rate over any configurable number of workspaces
"""
config = {
    # visualize
    "show_gui": False,
    "save_video": True,

    # paths:
    "maze_map_path": "TrainingNavigator/maps/bottleneck.png",
    "workspaces_path": 'TrainingNavigator/workspaces/bottleneckXL.npy',
    "stepper_agent_path": 'TrainingNavigator/StepperAgents/StepperV2_ep03_vel05_randInit.pt',
    "navigator_agent_path": "TrainingNavigator/logs/RandInitStepper_minV1.5_lessClipV2/saved_model/model_1100000.zip",
    "output_path": "TrainingNavigator/NavigatorTests",

    # Technical params
    "maze_size": (10, 10),
    "velocity_in_obs": False,
    "done_on_collision": False,
    "rewards": Rewards(target_arrival=1, collision=-1, fall=-1, idle=-0.001, ),
    "max_stepper_steps": 50,
    "max_navigator_steps": 20,
    "stepper_radius_range": (0.4, 2.5),
    "epsilon_to_subgoal": 0.35,
    "max_velocity_in_subgoal": 1.5,

    # logging parameters
    "eval_ws_num": 1000,  # will take the first workspaces

}


def evaluate_workspace(model):
    t_start = time.time()
    rewards = []
    episodes_length = []
    success_count = 0
    stats = pd.DataFrame([])

    for i in range(config["eval_ws_num"]):
        env_i = model.env.envs[0].env
        obs = env_i.reset(start_goal_pair_idx=i, create_video=True, video_path=video_path + f"_{i}.gif")
        steps, curr_reward, done = 0, 0, False

        while done is False:
            with torch.no_grad():
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env_i.step(action)
            curr_reward += reward
            steps += 1

        # collect info to sturct
        stats = stats.append(pd.DataFrame({k: [int(v)] for k, v in info.items()}))

        print(f"workspace-{i} done reason: {info}")
        rewards.append(curr_reward)
        episodes_length.append(steps)
        success_count += info['success']

    avg_reward = sum(rewards) / config["eval_ws_num"]
    avg_length = sum(episodes_length) / config["eval_ws_num"]
    success_rate = success_count / config["eval_ws_num"]

    print("All workspaces evaluation done in %.4f secs: " % (time.time() - t_start))
    return avg_reward, avg_length, success_rate, stats


if __name__ == '__main__':
    maze_map = blackwhiteswitch(config["maze_map_path"])
    start_goal_pairs = np.load(config["workspaces_path"]) / config["maze_size"][0]
    tested_nav_name = config["navigator_agent_path"].split('/')[-3]  # extract the examined navigator name

    # create dir for the specific tested navigator
    dir_name = os.path.join(config["output_path"], tested_nav_name + "_test")
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    video_path = os.path.join(dir_name, tested_nav_name)

    # create the maze environment
    eval_maze_env = MazeEnv(maze_size=config["maze_size"], maze_map=maze_map, xy_in_obs=True,
                            show_gui=config["show_gui"])
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

    # create the model
    DDPGMP_model = DDPGMP.load(config["navigator_agent_path"], env=eval_nav_env)
    avg_reward, avg_length, success_rate, stats = evaluate_workspace(model=DDPGMP_model)
    print(f"avg_length: {avg_length}")
    print(stats.describe())
