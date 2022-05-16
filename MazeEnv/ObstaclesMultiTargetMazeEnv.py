import numpy as np

from MazeEnv.MultiTargetMazeEnv import MultiTargetMazeEnv
from MazeEnv.EnvAttributes import MazeSize, Rewards


class ObstaclesMultiTargetMazeEnv(MultiTargetMazeEnv):
    def __init__(self, *args, **kwargs):
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

        # place obstacle within a radius from the goal, but not to close to starting point
        obstacle_x, obstacle_y = self._ant.start_position[:2]
        while np.linalg.norm(np.array([obstacle_x, obstacle_y]) - self._ant.start_position[:2]) < 1:
            goalx, goaly = self._target_loc[0], self._target_loc[1]
            obstacle_r = np.random.uniform(1, 2.5)
            obstacle_theta = np.random.uniform(0, 2 * np.pi)
            obstacle_x = goalx + obstacle_r * np.cos(obstacle_theta)
            obstacle_y = goaly + obstacle_r * np.sin(obstacle_theta)

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
