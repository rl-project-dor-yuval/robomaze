from wandb.wandb_torch import torch

from TrainingNavigator.NavigatorEnv import MultiWorkspaceNavigatorEnv
from TrainingNavigator.TD3MP import TD3MP
from TrainingNavigator.Utils import unscale_action


def test_workspace(actor, env: MultiWorkspaceNavigatorEnv, i):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor = actor.to(device)

    done = False
    obs = env.reset(workspace_idx=i)

    while not done:
        obs_torch = torch.from_numpy(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            action = actor(obs_torch)
            action = action.squeeze(0).to('cpu').numpy()

        action = unscale_action(env, action)

        obs, reward, done, info = env.step(action)

        if info['success']:
            return True

    return False


def test_navigator(actor: str, env: MultiWorkspaceNavigatorEnv):
    success_count = 0
    for i in range(env.workspace_count):
        success_count += test_workspace(actor, env, i)


def test_multiple_navigators(models_list: list[str], env: MultiWorkspaceNavigatorEnv):
    success_rates = []
    for model in models_list:
        actor = TD3MP.load(model, env=env).policy.actor.eval()
        success_rates.append(test_navigator(actor, env))

    return success_rates
