import torch


# TODO add support for device
class StepperAgent:
    """
     A wrapper for an agent, used to infer a trained actor network
    """
    def __init__(self, agent_path: str, device: str):
        if device == 'auto':
            _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            _device = torch.device(device)

        self.agent_nn = torch.load(agent_path, map_location=_device)
        self.agent_nn.eval()
        self.device = device

    def step(self, obs):
        obs = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            actions = self.agent_nn(obs)
            actions = actions.squeeze(0).numpy()

        return actions

