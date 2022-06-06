"""
Run multiple experiments given a base yaml file for configurations and a list of parameters
to override in each experiment (change here)
This is not generic yet and has to be edited inside the code
 """
import sys
sys.path.append('.')
import os
import subprocess
from MazeEnv.EnvAttributes import Rewards
import yaml

config_file = "TrainingNavigator/configs/BN_NoKillOnWall.yaml"

yaml_loader = yaml.Loader
yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
config = yaml.load(open(config_file, "r"), yaml_loader)

for seed in [42, 42**3]:  #, 42**3, 1948, 777777, 555555]:
    config["seed"] = seed
    config["project"] = "Robomaze-Tests"
    yaml.dump(config, open("TrainingNavigator/configs/temp.yaml", "w+"))
    subprocess.Popen("python TrainingNavigator/TrainingMultiproc.py temp.yaml", shell=True)
