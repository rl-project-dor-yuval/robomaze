import sys
sys.path.append('.')
import yaml
from MazeEnv.EnvAttributes import Rewards
from TrainingNavigator.NavigatorEnv import MultiWorkspaceNavigatorEnv
from TrainingNavigator.TestTrainedNavigators.NavAgents import TD3MPAgent, NavAgent, RRTAgent
from TrainingNavigator.TestTrainedNavigators.TestTrainedNavigator import test_multiple_navigators, get_env_from_config, \
    write_results_to_csv


if __name__ == '__main__':
    # test bottleneck:
    # best model is chosen manually in here, from next runs it will be "best_model.zip"
    bn_no_freespace_model = "TrainingNavigator/models/NewAnt_AllMazes/bottleneck_Ant_1308_093747_053031" \
                            "/saved_model/model_82000.zip"
    bn_with_freespace_model = "TrainingNavigator/models/NewAnt_AllMazes/bottleneck_with_fs/model_80000.zip"
    bn_no_demo_model = "TrainingNavigator/models/NewAnt_AllMazes/bottleneck_no_demo/model_72000.zip"

    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("TrainingNavigator/configs/BN_NoKillOnWall.yaml", "r"), yaml_loader)

    nav_env = get_env_from_config(config, robot="Ant")

    models = [bn_no_freespace_model, bn_with_freespace_model, bn_no_demo_model]
    demo_types = ['no_freespace', 'with_freespace', 'no_demo']

    agent_list: list[NavAgent] = [TD3MPAgent(model, nav_env, demo_t) for model, demo_t in zip(models, demo_types)]
    agent_list.append(RRTAgent(map_path=config['maze_map_path'], env=nav_env))

    success_rates, times = test_multiple_navigators(agent_list, nav_env)
    write_results_to_csv(agent_list, success_rates, times, config)

    # ------------------------
    # test S-narrow:
    # best model is chosen manually in here, from next runs it will be "best_model.zip"
    models = ["TrainingNavigator/models/NewAnt_AllMazes/S-narrow_no_demo/model_240000.zip",  # no demo
              "TrainingNavigator/models/NewAnt_AllMazes/S-narrow_with_freespace/model_225000.zip",  # with freespace"
              "TrainingNavigator/models/NewAnt_AllMazes/S-narrow_no_freespace/model_235000.zip"  # no freespace
              ]
    demo_types = ['no_demo', 'with_freespace', 'no_freespace']

    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("TrainingNavigator/configs/S-narrow.yaml", "r"), yaml_loader)

    nav_env = get_env_from_config(config, robot="Ant")
    agent_list: list[NavAgent] = [TD3MPAgent(model, nav_env, demo_t) for model, demo_t in zip(models, demo_types)]
    agent_list.append(RRTAgent(map_path=config['maze_map_path'], env=nav_env))

    success_rates, times = test_multiple_navigators(agent_list, nav_env)
    write_results_to_csv(agent_list, success_rates, times, config)

    # ------------------------
    # test 20x20maze
    models = ["TrainingNavigator/models/NewAnt_AllMazes/20x20maze_no_demo/model_330000.zip",  # no demo
              "TrainingNavigator/models/NewAnt_AllMazes/20x20maze_with_freespace/model_330000.zip",  # with freespace"
              "TrainingNavigator/models/NewAnt_AllMazes/20x20maze_no_freespace/model_380000.zip"  # no freespace
              ]
    demo_types = ['no_demo', 'with_freespace', 'no_freespace']

    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("TrainingNavigator/configs/20x20maze.yaml", "r"), yaml_loader)

    nav_env = get_env_from_config(config, robot="Ant")
    agent_list: list[NavAgent] = [TD3MPAgent(model, nav_env, demo_t) for model, demo_t in zip(models, demo_types)]
    agent_list.append(RRTAgent(map_path=config['maze_map_path'], env=nav_env))

    success_rates, times = test_multiple_navigators(agent_list, nav_env)
    write_results_to_csv(agent_list, success_rates, times, config)

    # ------------------------
    # test 20x20maze
    models = ["TrainingNavigator/models/NewAnt_AllMazes/spiral_no_demo/model_200000.zip",  # no demo
              "TrainingNavigator/models/NewAnt_AllMazes/spiral_with_freespace/model_320000.zip",  # with freespace"
              "TrainingNavigator/models/NewAnt_AllMazes/spiral_no_freespace/model_380000.zip"  # no freespace
              ]
    demo_types = ['no_demo', 'with_freespace', 'no_freespace']

    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("TrainingNavigator/configs/SpiralThick20x20.yaml", "r"), yaml_loader)

    nav_env = get_env_from_config(config, robot="Ant")
    agent_list: list[NavAgent] = [TD3MPAgent(model, nav_env, demo_t) for model, demo_t in zip(models, demo_types)]
    agent_list.append(RRTAgent(map_path=config['maze_map_path'], env=nav_env))

    success_rates, times = test_multiple_navigators(agent_list, nav_env)
    write_results_to_csv(agent_list, success_rates, times, config)

