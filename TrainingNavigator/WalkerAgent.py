import torch

# TODO add support for device
class WalkerAgent:
    """
     A wrapper for an agent, used to infer a trained actor network
    """
    def __init__(self, agent_path):
        self.agent_nn = torch.load(agent_path)
        self.agent_nn.eval()

    def step(self, obs):
        obs = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            actions = self.agent_nn(obs)
            actions = actions.squeeze(0).numpy()

        return actions
