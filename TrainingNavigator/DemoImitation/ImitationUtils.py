import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.data import TensorDataset, DataLoader

from MazeEnv.EnvAttributes import Workspace, Rewards
from TrainingNavigator.NavigatorEnv import MultiWorkspaceNavigatorEnv
from TrainingNavigator.TD3MP import CustomTD3Policy, TD3MP
from TrainingNavigator.Utils import trajectory_to_transitions_with_heading


def demos_to_tensor(demo_path: str, num_trajectories) -> (torch.Tensor, torch.Tensor):
    dummy_rewards = Rewards()
    demos = np.load(demo_path)
    observations_dataset = []
    actions_labels = []
    for i in range(len(demos)):
        traj_observations, traj_actions, _, _, _ = \
            trajectory_to_transitions_with_heading(demos[str(i)], dummy_rewards, 0.25)
        observations_dataset.append(np.array(traj_observations))
        actions_labels.append(np.array(traj_actions))
        if num_trajectories != 0 and i >= num_trajectories - 1:
            break

    observations_dataset = np.concatenate(observations_dataset)
    actions_labels = np.concatenate(actions_labels)

    return torch.Tensor(observations_dataset), torch.Tensor(actions_labels)


# def normalize_observations_tensor(obs, )


def get_policy_to_train(maze_size, learning_rate):
    """ to avoid bugs, create model and env just like normal training"""
    maze_env_kwargs = dict(maze_size=(maze_size), maze_map=np.zeros(maze_size), tile_size=1.,
                           show_gui="False", robot_type="Ant", )
    nav_env_kwargs = dict(workspace_list=[Workspace(0, 0, 0, 0, 0, 0)],
                          maze_env_kwargs=maze_env_kwargs,
                          epsilon_to_hit_subgoal=0.25,
                          stepper_agent="TrainingNavigator/StepperAgents/AntWithHeading.pt",
                          control_heading_at_subgoal=True, )

    # wrap env by vec env to make it compatible with TD3MP
    nav_env = make_vec_env(MultiWorkspaceNavigatorEnv, n_envs=1, env_kwargs=nav_env_kwargs)

    model = TD3MP(policy=CustomTD3Policy,
                  env=nav_env,
                  policy_kwargs=dict(net_arch=dict(pi=(400, 300), qf=(400, 300))),
                  learning_rate=learning_rate,
                  )

    return model.policy


def normalize_dataset(observations: torch.Tensor, actions: torch.Tensor, maze_size: tuple,
                      a_radius_low: float, a_radius_high: float):
    """ normalize observations and actions to [-1, 1]"""
    norm_obs = observations.clone()
    maze_size = torch.Tensor(maze_size)

    norm_obs[:, :2] = 2 * (norm_obs[:, :2] / maze_size) - 1
    norm_obs[:, 2] = norm_obs[:, 2] / np.pi
    norm_obs[:, 3:5] = 2 * (norm_obs[:, 3:5] / maze_size) - 1

    norm_actions = actions.clone()
    norm_actions[:, 0] = 2 * (norm_actions[:, 0] - a_radius_low) / (a_radius_high - a_radius_low) - 1
    norm_actions[:, 1:3] = norm_actions[:, 1:3] / np.pi

    return norm_obs, norm_actions


def prepare_data_loader(demo_traj_path,
                        batch_size,
                        device,
                        action_radius_high,
                        action_radius_low,
                        maze_size,
                        num_trajectories=0,
                        shuffle=True):
    """ prepare data loader for imitation learning. batch size of 0 means each batch is the whole dataset"""

    observations, actions = demos_to_tensor(demo_traj_path, num_trajectories)
    observations, actions = normalize_dataset(observations, actions, maze_size, action_radius_low, action_radius_high)
    observations = observations.to(device)
    actions = actions.to(device)

    if batch_size == 0:
        batch_size = len(observations)

    dataset = TensorDataset(observations, actions)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


def test_loss(actor, data_loader):
    """ test the loss of the model on the data loader"""
    loss = 0
    r_mean_err = 0
    next_direction_mean_err = 0
    next_heading_mean_err = 0
    for i, (obs, actions) in enumerate(data_loader):  # should do one iteration since batch size is all
        if i > 0:
            print("Warning: there are more than one batch in the data loader")
            break

        with torch.no_grad():
            actions_pred = actor(obs)
        err = actions_pred - actions
        r_mean_err += torch.mean(torch.abs(err[:, 0]))
        next_direction_mean_err += err[:, 1].mean()
        next_heading_mean_err += err[:, 2].mean()

        loss += torch.mean(err ** 2)

    return loss / len(data_loader), torch.abs(next_direction_mean_err) / len(data_loader), \
           torch.abs(next_heading_mean_err) / len(data_loader), torch.abs(r_mean_err) / len(data_loader)
