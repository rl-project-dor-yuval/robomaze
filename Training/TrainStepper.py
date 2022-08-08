import numpy as np
import sys
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
import wandb
from datetime import datetime

sys.path.append('.')
from MazeEnv.EnvAttributes import Rewards, Workspace
from Utils import get_multi_workspace_circle_envs, get_multi_targets_circle_envs_multiproc
from Evaluation import EvalAndSaveCallback, MultiWorkspaceEvalAndSaveCallback
import torch


def train_stepper(config: dict):
    # add datetime string to run name so each run will be on different dir in case we run a sweep:
    datetime_string = datetime.now().strftime("%d%m_%I%M%S_%f")
    config["run_name"] = config["run_name"] + datetime_string
    config["dir"] = "./Training/logs/StepperV3" + config["run_name"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("running on:", device)

    # setup W&B:
    wb_run = wandb.init(project="Stepper-all-robots", name=config["run_name"],
                        group=config["group"], config=config)
    wandb.tensorboard.patch(root_logdir="Training/logs/StepperV2/tb", pytorch=True)

    workspaces = np.genfromtxt("Training/workspaces/NoHeading/workspaces_06to3_train.csv", delimiter=',')
    # workspaces contains: (goal_x, goal_y) but should contain
    # (start_x, start_y, start_heading, goal_x, goal_y, goal_heading)
    # where start_xy and start_heading are constant in our case, and goal_heading is ignored
    # if we give high enough epsilon
    workspaces = np.concatenate((np.ones((workspaces.shape[0], 2)) * 5,
                                 np.zeros((workspaces.shape[0], 1)),
                                 workspaces,
                                 np.zeros((workspaces.shape[0], 1))), axis=1)
    workspaces = Workspace.list_from_multiple_arrays(workspaces)
    val_workspaces = np.genfromtxt("Training/workspaces/NoHeading/workspaces_06to3_validation.csv", delimiter=',')
    val_workspaces = np.concatenate((np.ones((val_workspaces.shape[0], 2)) * 5,
                                     np.zeros((val_workspaces.shape[0], 1)),
                                     val_workspaces,
                                     np.zeros((val_workspaces.shape[0], 1))), axis=1)
    val_workspaces = Workspace.list_from_multiple_arrays(val_workspaces)

    env_kwargs = dict(radius=config["map_radius"],
                      workspace_list=workspaces,
                      test_workspace_list=val_workspaces,
                      timeout_steps=config["timeout_steps"],
                      rewards=config["rewards"],
                      max_goal_velocity=config["max_goal_velocity"],
                      xy_in_obs=False,
                      show_gui=config["show_gui"],
                      hit_target_epsilon=config["target_epsilon"],
                      target_heading_epsilon=config["target_heading_epsilon"],
                      noisy_robot_initialization=config["random_initialization"],
                      with_obstacles=config["with_obstacles"],
                      sticky_actions=config["sticky_actions"],
                      success_steps_before_done=config["success_steps_before_done"],
                      done_on_goal_reached=config["done_on_goal_reached"],
                      robot_type=config["robot_type"], )
    if config["num_envs"] == 1:
        maze_env, eval_maze_env = get_multi_workspace_circle_envs(**env_kwargs)
    else:
        print("training stepper with multiple envrionments is inefficient."
              " if you want to do this anyway, remove this line and make sure "
              "get_multi_targets_circle_envs_multiproc is updated")
        exit(0)
        maze_env, eval_maze_env = get_multi_targets_circle_envs_multiproc(**env_kwargs,
                                                                          num_envs=config["num_envs"])

    callback = MultiWorkspaceEvalAndSaveCallback(log_dir=config["dir"],
                                                 eval_env=eval_maze_env,
                                                 eval_freq=config["eval_freq"],
                                                 eval_video_freq=config["video_freq"],
                                                 wb_run=wb_run,
                                                 verbose=1)

    # create_model
    exploration_noise = NormalActionNoise(mean=np.array([0] * eval_maze_env.action_space.shape[0]),
                                          sigma=np.array([config["exploration_noise_std"]] \
                                                         * eval_maze_env.action_space.shape[0]))

    def lr_func(progress):
        if progress < 0.33 and config["reduce_lr"]:
            return config["learning_rate"] * config["lr_reduce_factor"]
        else:
            return config["learning_rate"]

    model_kwargs = dict(policy="MlpPolicy",
                        env=maze_env,
                        buffer_size=config["buffer_size"],
                        learning_rate=lr_func,
                        batch_size=config["batch_size"],
                        action_noise=exploration_noise,
                        device=device,
                        train_freq=(1, "episode"),
                        verbose=0,
                        tensorboard_log="./Training/logs/StepperV2/tb",
                        learning_starts=config["learning_starts"],
                        gamma=config["gamma"],
                        seed=config["seed"],
                        tau=config["tau"], )
    if config["use_td3"]:
        model_kwargs["target_policy_noise"] = config["td3_smoothing_noise"]
        model_kwargs["target_noise_clip"] = config["td3_smoothing_noise_clip"]
        model_kwargs["policy_delay"] = config["td3_policy_delay"]
        model_kwargs["policy_kwargs"] = dict(n_critics=config["td3_n_critics"])
        model = TD3(**model_kwargs)
    else:
        model = DDPG(**model_kwargs)

    model.learn(total_timesteps=config["train_steps"], tb_log_name=config["run_name"], callback=callback)

    wb_run.finish()
