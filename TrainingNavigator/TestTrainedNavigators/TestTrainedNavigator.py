import csv
import time

import numpy as np
import yaml
import torch

from MazeEnv.EnvAttributes import Rewards
from TrainingNavigator.NavigatorEnv import MultiWorkspaceNavigatorEnv
from TrainingNavigator.TD3MP import TD3MP
from TrainingNavigator.TestTrainedNavigators.NavAgents import NavAgent
from TrainingNavigator.Utils import unscale_action, blackwhiteswitch, make_workspace_list
from TrainingNavigator.TrainCLI import stepper_agents_paths


def test_workspace(agent: NavAgent, env: MultiWorkspaceNavigatorEnv, i):
    done = False
    obs = env.reset(workspace_idx=i)

    while not done:
        action = agent(obs)
        action = unscale_action(env, action)

        obs, reward, done, info = env.step(action)

        if info['success']:
            return True

    return False


def test_navigator(agent: NavAgent, env: MultiWorkspaceNavigatorEnv):
    success_count = 0
    for i in range(env.workspace_count):
        success_count += test_workspace(agent, env, i)

    return success_count / env.workspace_count


def test_multiple_navigators(agent_list: list[NavAgent], env: MultiWorkspaceNavigatorEnv):
    success_rates = []
    eval_times = []
    for agent in agent_list:
        t = time.time()
        success_rates.append(test_navigator(agent, env))
        eval_times.append(time.time() - t)

    return success_rates, eval_times


def get_env_from_config(config: dict, robot):
    maze_map = blackwhiteswitch(config["maze_map_path"])
    # config fives us train workspaces, we want test workspaces!
    test_workspaces_path = config["workspaces_path"].replace("workspaces.npy", "test_workspaces.npy")
    workspaces = np.load(test_workspaces_path) / 10  # all maps granularity is 10
    workspaces = make_workspace_list(workspaces)

    # workspaces = workspaces[:3] # for testing purposes

    maze_env_kwargs = dict(maze_size=config["maze_size"], maze_map=maze_map, xy_in_obs=True,
                           show_gui=config["show_gui"], robot_type=robot )
    nav_env = MultiWorkspaceNavigatorEnv(workspace_list=workspaces,
                                         maze_env_kwargs=maze_env_kwargs,
                                         epsilon_to_hit_subgoal=config["epsilon_to_subgoal"],
                                         max_vel_in_subgoal=config["max_velocity_in_subgoal"],
                                         rewards=config["rewards"],
                                         done_on_collision=config["done_on_collision"],
                                         max_stepper_steps=config["max_stepper_steps"],
                                         max_steps=config["max_navigator_steps"],
                                         stepper_radius_range=config["stepper_radius_range"],
                                         stepper_agent=stepper_agents_paths[robot],
                                         wall_hit_limit=config["wall_hit_limit"],
                                         repeat_failed_ws_prob=0)

    return nav_env


def write_results_to_csv(agent_list: list[NavAgent], success_rates: list[float], eval_times: list[float], config: dict):
    with open("TrainingNavigator/TestTrainedNavigators/test_results.csv", "a") as f:
        writer = csv.writer(f)
        maze = config['maze_map_path'].split('/')[-1].split('.')[0]
        for agent, success_rate, eval_time in zip(agent_list, success_rates, eval_times):
            writer.writerow([agent.name, config['robot_type'], maze, agent.demo_type, success_rate, eval_time])
