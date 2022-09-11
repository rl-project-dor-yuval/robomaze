import numpy as np
import torch

from MazeEnv.EnvAttributes import Workspace, Rewards
from TrainingNavigator.DemoImitation.ImitationNN import ImitationNN
from TrainingNavigator.NavigatorEnv import MultiWorkspaceNavigatorEnv
from TrainingNavigator.TD3MP import CustomTD3Policy, TD3MP
from TrainingNavigator.Utils import trajectory_to_transitions_with_heading


def demos_to_tensor(demo_path: str) -> (torch.Tensor, torch.Tensor):
    demo_path = "./TrainingNavigator/workspaces/bottleneck/trajectories_train.npz"
    print("fix demo to use arg!!")

    dummy_rewards = Rewards()
    demos = np.load(demo_path)
    observations_dataset = []
    actions_labels = []
    for i in range(len(demos)):
        traj_observations, traj_actions, _, _, _= \
            trajectory_to_transitions_with_heading(demos[str(i)], dummy_rewards, 0.25)
        observations_dataset.append(np.array(traj_observations))
        actions_labels.append(np.array(traj_actions))

    observations_dataset = np.concatenate(observations_dataset)
    actions_labels = np.concatenate(actions_labels)

    # convert to tensor
    # normalize observations and actions:

    pass

def normalize_observations_tensor(obs, )


def get_policy_to_train():
    """ to avoid bugs, create model and env just like normal training"""
    maze_env_kwargs = dict(maze_size=(40, 40), maze_map=np.zeros((120, 120)), tile_size=1 / 3,
                           show_gui="False", robot_type="Ant", )
    nav_env = MultiWorkspaceNavigatorEnv(workspace_list=[Workspace(0, 0, 0, 0, 0, 0)],
                                         maze_env_kwargs=maze_env_kwargs,
                                         epsilon_to_hit_subgoal=0.25,
                                         stepper_agent="TrainingNavigator/StepperAgents/AntWithHeading.pt",
                                         control_heading_at_subgoal=True,)
    model = TD3MP(policy=CustomTD3Policy,
                  env=nav_env,
                  policy_kwargs=dict(net_arch=(400, 300), qf=(400, 300)))

    return model.policy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = get_policy_to_train().to(device)

    demoset_to_dataloader("")

    model = ImitationNN(5, 3)
    print(model)
    print(model(torch.rand(1, 3)))
