from stable_baselines3 import DDPG
import sys
import torch

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


