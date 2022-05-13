from pathlib import Path
import os, time, sys

sys.path.append('.')

import cv2
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Utils import blackwhiteswitch

from TrainingNavigator.NavigatorEnv import NavigatorEnv, MultiStartgoalNavigatorEnv
from TrainingNavigator.ExtractStepperAgent import extract_agent
from MazeEnv.MazeEnv import MazeEnv
from MazeEnv.EnvAttributes import Rewards
from DDPGMP import DDPGMP

"""
This script is used to track statistics of the trained navigator.
it shows the model's reasons for failure and success rate over any configurable number of workspaces
"""
config = {
    "name": "RandInitStepper_minV1.5_lessClipV2_doneOnCol",
    # if is None then the run name is extracted from nav agent path

    # visualize
    "show_gui": False,
    "save_video": True,

    # paths:
    "maze_map_path": "TrainingNavigator/maps/bottleneck.png",
    "workspaces_path": 'TrainingNavigator/workspaces/bottleneckXL.npy',
    "stepper_agent_path": 'TrainingNavigator/StepperAgents/StepperV2_ep03_vel05_randInit.pt',
    "navigator_agent_path": "TrainingNavigator/logs/RandInitStepper_minV1.5_lessClipV2/saved_model",  # Dir Required
    "output_path": "TrainingNavigator/NavigatorTests",

    # Technical params
    "maze_size": (10, 10),
    "velocity_in_obs": False,
    "done_on_collision": True,
    "rewards": Rewards(target_arrival=1, collision=-1, fall=-1, idle=-0.001, ),
    "max_stepper_steps": 50,
    "max_navigator_steps": 20,
    "stepper_radius_range": (0.4, 2.5),
    "epsilon_to_subgoal": 0.35,
    "max_velocity_in_subgoal": 1.5,

    # logging parameters
    "eval_ws_num": 1000,  # will take the first workspaces

}


def play_workspace(env, model, idx, create_video=False):
    steps, curr_reward, done = 0, 0, False
    obs = env.reset(start_goal_pair_idx=idx, create_video=create_video, video_path=video_path + f"_{idx}.gif")

    while done is False:
        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        curr_reward += reward
        steps += 1

    return info, steps


def evaluate_workspace(model, env):
    t_start = time.time()
    episodes_length = []
    success_count = 0
    stats = pd.DataFrame({'success': [0],
                          'fell': [0],
                          'hit_maze': [0],
                          'navigator_timeout': [0],
                          'stepper_timeout': [0]})

    for i in range(config["eval_ws_num"]):
        # env_i = model.env.envs[0].env
        env_i = env
        info, steps = play_workspace(env_i, model, i, create_video=False)

        # collect info to sturct
        if not config["done_on_collision"]:
            # ignore hit maze
            info["hit_maze"] = 0

        reason = [k for k, v in info.items() if v is True][0]
        stats[reason] += 1

        # if episode failed so record with video
        if info["success"] is False:
            play_workspace(env_i, model, i, create_video=True)

        print(f"workspace-{i} done reason: {reason}")
        episodes_length.append(steps)
        success_count += info['success']

    avg_length = sum(episodes_length) / config["eval_ws_num"]
    success_rate = success_count / config["eval_ws_num"]

    print("All workspaces evaluation done in %.4f secs: " % (time.time() - t_start))
    return avg_length, success_rate, stats


if __name__ == '__main__':
    maze_map = blackwhiteswitch(config["maze_map_path"])
    start_goal_pairs = np.load(config["workspaces_path"]) / config["maze_size"][0]
    # extract the examined navigator name
    tested_nav_name = config["name"] if config["name"] else config["navigator_agent_path"].split('/')[-3]
    print(tested_nav_name)
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
    # if there is a file ends with pt, then load the model
    # else if there are multiple files end with .zip take the last one, and load the extract the model
    # from it, otherwise raise excpetion

    model_path = os.path.join(config["navigator_agent_path"], "NavAgent.pt")
    if os.path.isfile(model_path):
        model = torch.load(model_path)
    elif os.listdir(config["navigator_agent_path"]):
        model_path = os.path.join(config["navigator_agent_path"],
                                  max(os.listdir(config["navigator_agent_path"]),
                                      key=lambda x: int(x.split(".")[0].split("_")[1])))
        model = extract_agent(model_path, save_path=os.path.join(config["navigator_agent_path"], "NavAgent.pt"))
    else:
        raise FileNotFoundError("No model found in the path")

    avg_length, success_rate, stats = evaluate_workspace(model=model, env=eval_nav_env)

    print(f"avg_length: {avg_length}")
    # get index list from dictionary

    print("saving stats bar plot")
    # plot the stats as the key in vertical axis and value is in horizontal axis
    keys = list(stats.keys())
    values = stats.values[0]
    sns.barplot(x=values, y=keys)
    plt.savefig(os.path.join(dir_name, "stats_bar_plot.png"))
