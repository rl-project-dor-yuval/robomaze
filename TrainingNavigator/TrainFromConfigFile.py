"""
Usage: TrainingFromConfigFile.py ConfigFileName.yaml
the config file must appear in TrainingNavigator/Configs and you shouldn't pass the path
"""

import sys
sys.path.append('.')
from MazeEnv.EnvAttributes import Rewards
import yaml
import argparse
from TrainingNavigator.TrainingMultiproc import train_multiproc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', type=str, help='config file name without path,'
                                                      ' the file must appear in TrainingNavigator/configs/')
    args = parser.parse_args()

    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("TrainingNavigator/configs/" + args.config_name, "r"), yaml_loader)

    train_multiproc(config)
