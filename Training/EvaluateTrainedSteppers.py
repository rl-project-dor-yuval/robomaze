""" evaluate trained steppers from a given list of checkpoints on the train set, and not just for the success rate.
    we dont use random initialization here because we want to evaluate the trained steppers equally."""

import os, sys
sys.path.append('../..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import MazeEnv.MultWorkspaceMazeEnv as mz
from Training.Utils import make_circular_map
from TrainingNavigator.StepperAgent import StepperAgent
from MazeEnv.EnvAttributes import Rewards


num_workspaces = 100


def get_env(sticky_actions, timeout_steps):
    target_list = np.genfromtxt("Training/workspaces/goals_06to3_test.csv", delimiter=',')
    tile_size = 0.1
    maze_size = mz.MazeSize.SQUARE10
    map_size = np.dot(maze_size, int(1 / tile_size))
    maze_map = make_circular_map(map_size, 5 / tile_size)

    rewards = Rewards(target_distance_weight=0.01, rotation_weight=0.01, target_distance_offset=5, fall=-1,
                      target_arrival=1, collision=0, timeout=0, idle=0,)

    raise NotImplementedError("Fix to new workspaces")
    maze_env = mz.MultiTargetMazeEnv(maze_size=maze_size,
                                     maze_map=maze_map,
                                     tile_size=tile_size,
                                     start_loc=(5, 5),
                                     target_loc_list=target_list[:, :2],
                                     target_heading_list=target_list[:, 2],
                                     rewards=rewards,
                                     hit_target_epsilon=0.25,
                                     target_heading_epsilon=np.pi/9,
                                     done_on_goal_reached=False,
                                     timeout_steps=timeout_steps,
                                     sticky_actions=sticky_actions,
                                     max_goal_velocity=9999,
                                     show_gui=False,
                                     xy_in_obs=False,
                                     noisy_ant_initialization=False)

    return maze_env


def evaluate_stepper(agent, env: mz.MultiWorkspaceMazeEnv):
    fall_count = 0
    last_step_success_count = 0
    ep_success_count_list = []
    episodes_length_list = []

    for i in range(num_workspaces):
        obs = env.reset(target_index=i)
        step_count = 0
        ep_success_steps_count = 0

        done = False
        while not done:
            action = agent.step(obs)
            obs, reward, done, info = env.step(action)

            step_count += 1
            if info['success']:
                ep_success_steps_count += 1
            if info['fell']:
                fall_count += 1

        last_step_success_count += info['success']
        ep_success_count_list.append(ep_success_steps_count)
        episodes_length_list.append(step_count)

    episodes_with_success = np.sum(np.array(ep_success_count_list) > 0)

    success_rate_once = float(episodes_with_success) / float(num_workspaces)
    success_rate_last_step = float(last_step_success_count) / float(num_workspaces)
    fall_rate = float(fall_count) / float(num_workspaces)
    mean_episodes_length = np.mean(np.array(episodes_length_list))

    return success_rate_once, success_rate_last_step, fall_rate, mean_episodes_length


if __name__ == "__main__":

    # load list of checkpoints from chosen models:
    log_dirs = ["Training/logs/StepperV2--fixed--stickyActions8_max_steps_150"]
    stepper_checkpoints = []

    # for log_dir in log_dirs:
    #     # choose every milionth checkpoint
    #     stepper_checkpoints += glob.glob(log_dir + "/model_*000000.zip")
    # remove checkpoints before step 5 milion:
    # for c in stepper_checkpoints:
    #     if int(c.split("_")[-1].split(".")[0]) < 5000000:
    #         stepper_checkpoints.remove(c)

    # stepper_checkpoints += glob.glob(log_dirs[0] + "/model_17600000.zip")
    # stepper_checkpoints += glob.glob(log_dirs[0] + "/model_15000000.zip")
    stepper_checkpoints += glob.glob(log_dirs[0] + "/model_25000000.zip")

    env = get_env(sticky_actions=8, timeout_steps=150)

    run_name_list = []
    success_once_list = []
    success_last_list = []
    fall_rate_list = []
    ep_len_list = []

    for model_path in stepper_checkpoints:
        # load agents. some files got messed up downloading from other computer so we skip them:
        try:
            agent = StepperAgent(model_path)
        except:
            continue

        print(f'-----------------------')
        print(f'evaluating {model_path}')

        success_rate_once, success_rate_last_step, fall_rate, mean_episodes_length = \
            evaluate_stepper(agent, env)
        # filter poor results in advance:
        if success_rate_once < 0.85 or fall_rate > 0.05:
            continue
        print(f'saving results')

        run_name_list.append(model_path.split("V2")[-1])
        success_once_list.append(success_rate_once)
        success_last_list.append(success_rate_last_step)
        fall_rate_list.append(fall_rate)
        ep_len_list.append(mean_episodes_length)

    results = [run_name_list, success_once_list, success_last_list, fall_rate_list, ep_len_list]
    df = pd.DataFrame(results).T
    df.set_axis(['run_name', 'success_rate_once', 'success_rate_last_step', 'fall_rate', 'mean_episodes_length'], axis=1, inplace=True)
    df.set_index('run_name', inplace=True)
    df.plot.bar(y=['success_rate_once', 'success_rate_last_step', 'fall_rate'], rot=0)
    plt.xticks(fontsize=5)
    plt.show()
    df.to_csv('Training/trained_stepper_results.csv')




