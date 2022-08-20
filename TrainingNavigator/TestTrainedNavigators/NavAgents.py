""" define classes for nav agents to test in the test loop"""

from abc import abstractmethod
import torch
import numpy as np

from TrainingNavigator.NavigatorEnv import NavigatorEnv
from TrainingNavigator.TD3MP import TD3MP
from TrainingNavigator.TrajectoryGenerator import TrajGenerator
from TrainingNavigator.Utils import scale_action


class NavAgent:

    name: str
    demo_type: str

    @abstractmethod
    def __call__(self, obs):
        # return action
        pass


class TD3MPAgent(NavAgent):
    def __init__(self, model_path, env: NavigatorEnv, demo_type='no_demo'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("using device: ", self.device)
        self.policy = TD3MP.load(model_path, env=env).policy.actor.eval().to(self.device)
        self.name = (model_path.split('logs'))[-1]
        self.demo_type = demo_type

    def __call__(self, obs):
        obs_torch = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.policy(obs_torch)
            action = action.squeeze(0).to('cpu').numpy()
        return action


class RRTAgent(NavAgent):
    def __init__(self, map_path, env: NavigatorEnv):
        self.name = 'RRTAgent'
        self.demo_type = 'RRTAgent-just_demo'

        self.env = env
        self.trajectory_generator = TrajGenerator(map_path)

    def __call__(self, obs):
        obs = self.env.unormalize_obs_if_needed(obs)
        # map granularity is 10, and trajectory generator accepts tuples and not arrays and uses coodinates that
        # start from bottom left.
        # move to trajectory generator coordinates:
        start = obs[:2] * 10
        start_traj_coords = (start[1], self.trajectory_generator.map.shape[0] - start[0] - 1)
        goal = obs[3:5] * 10
        goal_traj_coords = (goal[1], self.trajectory_generator.map.shape[0] - goal[0] - 1)

        trajectory = self.trajectory_generator.find_optimal_trajectories(start_traj_coords,
                                                                         goal_traj_coords,
                                                                         numOfTrajs=1,
                                                                         plot=False)[0]

        # for some reason we get the back in normal coordinates

        action_xy = np.array(trajectory[1]) - np.array(trajectory[0])
        action_xy = action_xy * 0.1  # map granularity is 10

        r = np.linalg.norm(action_xy)
        theta = np.arctan2(action_xy[1], action_xy[0])

        # need to scale action to work as TD3MP agent
        return scale_action(self.env, np.array([r, theta]))

