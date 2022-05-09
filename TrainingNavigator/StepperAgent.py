import torch
from stable_baselines3 import DDPG

class StepperAgent:
    """
     A wrapper for an agent, used to infer a trained actor network
    """
    def __init__(self, agent_path: str, device: str = 'auto'):
        if device == 'auto':
            _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            _device = torch.device(device)

        # agentpath could be pt file or zip file from SB3 format
        if agent_path.split(".")[-1] == 'pt':
            self.agent_nn = torch.load(agent_path, map_location=_device)
        else:# zip file
            model = DDPG.load(agent_path)
            self.agent_nn = model.policy.actor

        self.agent_nn.eval()
        self.device = _device

    def step(self, obs):
        obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            actions = self.agent_nn(obs)
            actions = actions.squeeze(0).to('cpu').numpy()

        return actions
