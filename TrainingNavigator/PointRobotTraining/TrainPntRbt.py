import time

from MazeEnv.EnvAttributes import Rewards
from PointRobotEnv import PointRobotEnv

# Setup Training Environment
import cv2
import numpy as np

# --- Parameters
config = {
    "run_name": "PntRbt_1",
    "show_gui": True,
    "seed": 42 ** 3,
    "train_steps": 5 * 10 ** 2,

    # Training and environment parameters
    "learning_rate": 0.5e-5,
    "grad_clip_norm_actor": 3,
    "grad_clip_norm_critic": 0.5,
    "batch_size": 2048,
    "buffer_size": 2 * 10 ** 5,
    "actor_arch": [64, 64],  # Should not be changed or explored
    "critic_arch": [64, 64],  # Should not be changed or explored
    "exploration_noise_std": 0.03,
    "epsilon_to_subgoal": 0.8,  # DO NOT TOUCH
    "stepper_radius_range": (1, 2.5),
    "done_on_collision": True,  # modify rewards in case you change this
    "rewards": Rewards(target_arrival=1, collision=-1, fall=-1, idle=-0.005, ),
    "demonstration_path": 'TrainingNavigator/workspaces/bottleneckXL_short1.5_trajectories.npz',
    "demo_on_fail_prob": 0.5,
    "learning_starts": 10 ** 4,

    # "max_stepper_steps": 75,
    "max_navigator_steps": 100,

    # logging parameters
    "eval_workspaces": 100,  # will take the first workspaces
    "eval_freq": 20000,
    "video_freq": 1,
    "save_model_freq": 50000,

    # Constants:
    "maze_size": (10, 10)
}

maze_map = - (cv2.imread('TrainingNavigator/maps/bottleneck.png', cv2.IMREAD_GRAYSCALE) / 255) + 1
maze_map = maze_map.T

start_goal_pairs = np.load('TrainingNavigator/workspaces/bottleneckXL.npy') / config["maze_size"][0]

pntEnv = PointRobotEnv(max_steps=config["max_navigator_steps"],
                       rewards=config["rewards"],
                       radius_range=config["stepper_radius_range"],
                       done_on_collision=config["done_on_collision"],
                       epsilon_to_hit_subgoal=config["epsilon_to_subgoal"],
                       start_goal_pairs=start_goal_pairs,
                       maze_map=maze_map,
                       visualize=config["show_gui"])

pntEnv.reset(start_goal_pair_idx=0)
start = time.time()

for i in range(50000):
    user_action = input("Enter radius, theta as (R,Theta) in degrees R between 1 and 2.5: ")
    nxt_act = np.array([float(x) for x in user_action.split(",")], dtype=np.float32)
    nxt_act[1] = np.float32(np.radians(nxt_act[1]))

    obs, reward, is_done, _ = pntEnv.step(nxt_act)
    if is_done:
        pntEnv.reset()
        print(f"Reward: {reward}")
    time.sleep(1. / 100)

print(time.time() - start)
