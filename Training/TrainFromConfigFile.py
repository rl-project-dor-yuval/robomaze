"""
Usage: TrainStepper.py ConfigFileName.yaml
the config file must appear in Training/Configs and you shouldn't pass the full path
"""
import sys
sys.path.append('.')
import yaml
import argparse
from MazeEnv.EnvAttributes import Rewards
from Training.TrainStepper import train_stepper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', type=str, help='config file name without path,'
                                                      ' the file must appear in Training/configs/')
    args = parser.parse_args()

    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("Training/configs/" + args.config_name, "r"), yaml_loader)

    train_stepper(config)