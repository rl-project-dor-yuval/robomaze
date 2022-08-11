import sys
sys.path.append('.')
import yaml
import argparse
from MazeEnv.EnvAttributes import Rewards
from datetime import datetime
from TrainingNavigator.TrainingMultiproc import train_multiproc


maps_config_file = {'bottleneck': 'TrainingNavigator/configs/BN_NoKillOnWall.yaml',
                    'S-narrow': 'TrainingNavigator/configs/S-narrow.yaml',
                    'SpiralThick20x20': 'TrainingNavigator/configs/SpiralThick20x20.yaml',
                    '20x20maze': 'TrainingNavigator/configs/20x20maze.yaml',
                    }

stepper_agents_paths = {'Ant': 'TrainingNavigator/StepperAgents/AntNoHeading.pt',
                        'Rex': 'TrainingNavigator/StepperAgents/RexNoHeading.pt',
                        }

if __name__ == '__main__':
    # load BN config just for keys:
    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("TrainingNavigator/configs/BN_NoKillOnWall.yaml", "r"), yaml_loader)

    # add all config parameters to argparse:
    parser = argparse.ArgumentParser()
    for key in config:
        parser.add_argument('--' + key, type=type(config[key]), default=config[key])
    # add rewards seperately:
    reward_dict = vars(config["rewards"])
    for rew in reward_dict.keys():
        parser.add_argument('--reward_' + rew, type=type(reward_dict[rew]), default=reward_dict[rew])

    # add a mandatory arguments for map, robot and demo kind:
    parser.add_argument('--map', choices=['bottleneck', 'S-narrow', 'SpiralThick20x20', '20x20maze'],
                        default='bottleneck', required=True)
    parser.add_argument('--robot', type=str, default='Ant', required=True)
    parser.add_argument('--demo_kind', choices=['no_demo', 'with_freespace', 'no_freespace'], default='no_freespace',
                        required=True)

    args = parser.parse_args()

    # get real config:
    config = yaml.load(open(maps_config_file[args.map], "r"), yaml_loader)

    # update config with relevant args:
    for key in args.__dict__:
        # handle rewards separately:
        if key.startswith("reward_"):
            reward_name = key.split("_")[1]
            setattr(config["rewards"], reward_name, args.__dict__[key])
        # all other parameters:
        elif key in config.keys():
            config[key] = args.__dict__[key]

    # change robot and demo kind according to args:
    config['stepper_agent_path'] = stepper_agents_paths[args.robot]
    config['robot_type'] = args.robot

    # by default, configs use no_freespace. If we want to use freespace or no demo we need to change the config:
    if args.demo_kind == 'no_demo':
        config['demo_on_fail_prob'] = 0.0
    elif args.demo_kind == 'with_freespace':
        config['demonstration_path'] = config['demonstration_path'].replace('_no_freespace', '')
        config['validation_demonstration_path'] = config['validation_demonstration_path'].replace('_no_freespace', '')

    config['tags'] = [args.map, args.robot, args.demo_kind]
    datetime_string = datetime.now().strftime("%d%m_%I%M%S_%f")
    config['run_name'] = args.map + '_' + args.robot + '_' + datetime_string

    train_multiproc(config)


