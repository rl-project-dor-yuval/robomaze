import numpy as np

from MazeEnv.MultWorkspaceMazeEnv import MultiWorkspaceMazeEnv
from MazeEnv.EnvAttributes import MazeSize, Rewards


class ObstaclesMultiTargetMazeEnv(MultiWorkspaceMazeEnv):

    min_ant_goal_distance_to_place_obstacle = 1.5
    obstacle_radius_random_offset_range = 0.1
    obstacle_angle_random_offset_range = np.pi/4

    def __init__(self, *args, **kwargs):
        raise DeprecationWarning("ObstaclesMultiTargetMazeEnv is has to be updated to support target heading?")
        super().__init__(*args, **kwargs)
        self.obstacle_uid = None

    def reset(self, with_obstacle=True, create_video=False, video_path=None, reset_episode_count=False,
              target_index=None):
        obs = super().reset(create_video, video_path, reset_episode_count)

        # remove old obstacle if there is one:
        if self.obstacle_uid is not None:
            self._pclient.removeBody(self.obstacle_uid)
            self.obstacle_uid = None

        if with_obstacle:
            self.obstacle_uid = self._place_obstacle()

        return obs

    def _place_obstacle(self):
        obstacle_type = np.random.choice(['pole', 'box', 'nothing'])
        if obstacle_type == 'nothing':
            return None

        ant_x, ant_y = self._ant.start_position[:2]
        goal_x, goal_y = self._target_loc[0], self._target_loc[1]

        goal_dist = np.linalg.norm([ant_x - goal_x, ant_y - goal_y])
        if np.linalg.norm([ant_x - goal_x, ant_y - goal_y]) < self.min_ant_goal_distance_to_place_obstacle:
            # no room for an obstacle, give up for this time
            return None

        # place obstacle between the ant and the goal with a random angle offset
        obstacle_r = goal_dist / 2 + np.random.uniform(-self.obstacle_radius_random_offset_range,
                                                       self.obstacle_radius_random_offset_range)
        goal_angle = np.arctan2(goal_y - ant_y, goal_x - ant_x)
        obstacle_angle = goal_angle + np.random.uniform(-self.obstacle_angle_random_offset_range,
                                                        self.obstacle_angle_random_offset_range)

        obstacle_x = ant_x + obstacle_r * np.cos(obstacle_angle)
        obstacle_y = ant_y + obstacle_r * np.sin(obstacle_angle)

        if obstacle_type == 'pole':
            uid = self._pclient.loadURDF("pole_obstacle.urdf",
                                         basePosition=(obstacle_x, obstacle_y, 0))
        elif obstacle_type == 'box':
            box_angle = np.random.uniform(0, 2 * np.pi)
            box_orientation = self._pclient.getQuaternionFromEuler([0, 0, box_angle])
            uid = self._pclient.loadURDF("box_obstacle.urdf",
                                         basePosition=(obstacle_x, obstacle_y, 0),
                                         baseOrientation=box_orientation)
        return uid
