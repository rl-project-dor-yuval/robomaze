import sys
sys.path.append('.')
import yaml
from MazeEnv.EnvAttributes import Rewards
from TrainingNavigator.NavigatorEnv import MultiWorkspaceNavigatorEnv
from TrainingNavigator.TestTrainedNavigators.NavAgents import TD3MPAgent, NavAgent, RRTAgent
from TrainingNavigator.TestTrainedNavigators.TestTrainedNavigator import test_multiple_navigators, get_env_from_config, \
    write_results_to_csv


if __name__ == '__main__':
    dir = "TrainingNavigator/models/AntWithHeading/"
    robot = "Ant"
    demo_types = ['no_demo', 'with_freespace', 'no_freespace']

    # test bottleneck:
    # best model is chosen manually in here, from next runs it will be "best_model.zip"
    models = [dir + "/bottleneck_" + robot + "_" + demo_t + "/saved_model/" + "best_model.zip"
              for demo_t in demo_types]

    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("TrainingNavigator/configs/BN_NoKillOnWall.yaml", "r"), yaml_loader)

    nav_env = get_env_from_config(config, robot=robot)


    agent_list: list[NavAgent] = [TD3MPAgent(model, nav_env, demo_t) for model, demo_t in zip(models, demo_types)]
    agent_list.append(RRTAgent(map_path=config['maze_map_path'], env=nav_env))

    success_rates, times = test_multiple_navigators(agent_list, nav_env)
    print("writing results to csv")
    write_results_to_csv(agent_list, success_rates, times, config)

    # ------------------------
    # test S-narrow:
    # best model is chosen manually in here, from next runs it will be "best_model.zip"
    models = [dir + "/S-narrow_" + robot + "_" + demo_t + "/saved_model/" + "best_model.zip"
              for demo_t in demo_types]

    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("TrainingNavigator/configs/S-narrow.yaml", "r"), yaml_loader)

    nav_env = get_env_from_config(config, robot=robot)
    agent_list: list[NavAgent] = [TD3MPAgent(model, nav_env, demo_t) for model, demo_t in zip(models, demo_types)]
    agent_list.append(RRTAgent(map_path=config['maze_map_path'], env=nav_env))

    success_rates, times = test_multiple_navigators(agent_list, nav_env)
    print("writing results to csv")
    write_results_to_csv(agent_list, success_rates, times, config)

    # ------------------------
    # test 20x20maze
    models = [dir + "/20x20maze_" + robot + "_" + demo_t + "/saved_model/" + "best_model.zip"
              for demo_t in demo_types]

    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("TrainingNavigator/configs/20x20maze.yaml", "r"), yaml_loader)

    nav_env = get_env_from_config(config, robot=robot)
    agent_list: list[NavAgent] = [TD3MPAgent(model, nav_env, demo_t) for model, demo_t in zip(models, demo_types)]
    agent_list.append(RRTAgent(map_path=config['maze_map_path'], env=nav_env))

    success_rates, times = test_multiple_navigators(agent_list, nav_env)
    print("writing results to csv")
    write_results_to_csv(agent_list, success_rates, times, config)

    # ------------------------
    # test Spiral
    models = [dir + "/SpiralThick20x20_" + robot + "_" + demo_t + "/saved_model/" + "best_model.zip"
              for demo_t in demo_types]

    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("TrainingNavigator/configs/SpiralThick20x20.yaml", "r"), yaml_loader)

    nav_env = get_env_from_config(config, robot=robot)
    agent_list: list[NavAgent] = [TD3MPAgent(model, nav_env, demo_t) for model, demo_t in zip(models, demo_types)]
    agent_list.append(RRTAgent(map_path=config['maze_map_path'], env=nav_env))

    success_rates, times = test_multiple_navigators(agent_list, nav_env)
    print("writing results to csv")
    write_results_to_csv(agent_list, success_rates, times, config)

    # ------------------------
    # test room10x10
    models = [dir + "/room10x10_" + robot + "_" + demo_t + "/saved_model/" + "best_model.zip"
              for demo_t in demo_types]

    yaml_loader = yaml.Loader
    yaml_loader.add_constructor("!Rewards", Rewards.from_yaml)
    config = yaml.load(open("TrainingNavigator/configs/room10x10.yaml", "r"), yaml_loader)

    nav_env = get_env_from_config(config, robot=robot)
    agent_list: list[NavAgent] = [TD3MPAgent(model, nav_env, demo_t) for model, demo_t in zip(models, demo_types)]
    agent_list.append(RRTAgent(map_path=config['maze_map_path'], env=nav_env))

    success_rates, times = test_multiple_navigators(agent_list, nav_env)
    print("writing results to csv")
    write_results_to_csv(agent_list, success_rates, times, config)

