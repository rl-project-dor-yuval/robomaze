"""
for usage can copy from here:
python TrainingNavigator/ExtractStepperAgent.py Training/logs/StepperV2LongRun_BS400_harder/best_model.zip TrainingNavigator/StepperAgents/StepperV2_ep03_vel05.pt
"""

from stable_baselines3 import DDPG
import sys
import torch
sys.path.append('.')

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: ")
        print(">python ExtractStepperAgent.py <path to model> [save path]")
        print("if a save path is not specified, the agent is saved in the"
              " current directory as StepperAgent.pt")
        exit()

    model_path = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) == 3 else "StepperAgent.pt"

    model = DDPG.load(model_path)
    actor = model.policy.actor
    torch.save(actor, save_path)


def extract_agent(model_path, save_path):
    assert model_path.endswith(".zip"), "model_path must end with .zip"
    assert save_path.endswith(".pt"), "save_path must be provided"

    model = DDPG.load(model_path)
    actor = model.policy.actor
    torch.save(actor, save_path)
    return actor
