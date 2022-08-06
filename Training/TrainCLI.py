import sys
sys.path.append('.')
import yaml
import argparse
from MazeEnv.EnvAttributes import Rewards
from Training.TrainStepper import train_stepper

if __name__ == '__main__':
    # load base config:
    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("Training/configs/BaseParams.yaml", "r"), yaml_loader)

    # add all config parameters to argparse:
    parser = argparse.ArgumentParser()
    for key in config:
        parser.add_argument('--' + key, type=type(config[key]), default=config[key])
    # add rewards seperately:
    reward_dict = vars(config["rewards"])
    for rew in reward_dict.keys():
        parser.add_argument('--reward_' + rew, type=type(reward_dict[rew]), default=reward_dict[rew])
    args = parser.parse_args()

    # update config with args:
    for key in args.__dict__:
        # handle rewards separately:
        if key.startswith("reward_"):
            reward_name = key.split("_")[1]
            setattr(config["rewards"], reward_name, args.__dict__[key])
        # all other parameters:
        else:
            config[key] = args.__dict__[key]

    # profit:
    print("training with config:", config)
    print("rewards:", vars(config['rewards']))
    train_stepper(config)
