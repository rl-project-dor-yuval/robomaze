import numpy as np
import glob
import os
import ipyplot

import MazeEnv.MultiTargetMazeEnv as mtmz
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DDPG
from Training.Evaluation import create_gifs_from_avi


def clear_files(path: str):
    """
    clean the files from path - accepts file patterns as well
    """
    files = glob.glob(path)
    for f in files:
        os.remove(f)


def make_circular_map(size, radius):
    """
    :param size : size of the map
    :param radius : radius of the maze

    :return : bitmap numpy array of the maze to push to MazeEnv
    """
    center = np.divide(size, 2)
    x, y = np.ogrid[:size[0], :size[1]]
    maze_map = np.where(np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) > radius, 1, 0)

    return maze_map


def get_multi_targets_circle_envs(radius, targets, timeout_steps, rewards, xy_in_obs, show_gui):
    # create environment :
    tile_size = 0.1
    maze_size = mtmz.MazeSize.SQUARE10
    map_size = np.dot(maze_size, int(1 / tile_size))
    circle_radius = radius
    maze_map = make_circular_map(map_size, circle_radius / tile_size)
    start_loc = (5, 5)

    maze_env = mtmz.MultiTargetMazeEnv(maze_size=maze_size,
                                       maze_map=maze_map,
                                       tile_size=tile_size,
                                       start_loc=start_loc,
                                       target_loc_list=targets,
                                       timeout_steps=timeout_steps,
                                       show_gui=show_gui,
                                       rewards=rewards,
                                       xy_in_obs=xy_in_obs)
    maze_env = Monitor(maze_env)

    check_env(maze_env)

    # create separete evaluation environment:
    eval_maze_env = mtmz.MultiTargetMazeEnv(maze_size=maze_size,
                                            maze_map=maze_map,
                                            tile_size=tile_size,
                                            start_loc=start_loc,
                                            target_loc_list=targets,
                                            timeout_steps=timeout_steps,
                                            show_gui=False,
                                            rewards=rewards,
                                            xy_in_obs=xy_in_obs)
    return maze_env, eval_maze_env


def visualize_model(model_path, targets_n, video_dir, eval_env: mtmz):
    """
    evaluate trained model and make videos of it on targets_n targets.
    
    :param model_path : path to model (best_model.zip file)
    :param targets_n : number of targets visualized
    :param video_dir : where the videos will be saved - directory
    :param eval_env : Evaluation environment of type MazeEnv.MultiTargetsMazeEnv

    """
    rewards = []
    reach_target_count = 0

    model = DDPG.load(os.path.join(model_path, "best_model"))

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    clear_files(os.path.join(video_dir, '*'))

    for tgt in range(targets_n):
        episode_reward = 0
        done = False

        full_video_path = os.path.join(video_dir, "final" + str(tgt) + ".avi")
        print(full_video_path)

        obs = eval_env.reset(create_video=True, video_path=full_video_path, target_index=tgt)
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            episode_reward += reward

        rewards.append("reward: " + str(episode_reward))
        if episode_reward > 0:
            reach_target_count += 1

    _ = eval_env.reset()

    create_gifs_from_avi(video_dir)
    print("reached target:", reach_target_count)

    gifs = glob.glob(os.path.join(video_dir, "*.gif"))
    gifs.sort()
    labels = [pth.split('/')[-1].split('.')[0] for pth in gifs]
    labels.sort()
    ipyplot.plot_images(gifs, labels, rewards, img_width=200)
