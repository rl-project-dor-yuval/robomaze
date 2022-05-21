import numpy as np

from MazeEnv.MazeEnv import MazeEnv
from MazeEnv.EnvAttributes import MazeSize, Rewards


class MultiTargetMazeEnv(MazeEnv):
    def __init__(self,
                 maze_size=MazeSize.MEDIUM,
                 maze_map: np.ndarray = None,
                 tile_size=0.1,
                 start_loc=(1, 1),
                 target_loc_list=[(3, 3)],
                 rewards: Rewards = Rewards(),
                 timeout_steps: int = 0,
                 show_gui: bool = False,
                 xy_in_obs: bool = True,
                 hit_target_epsilon: float = 0.8,
                 done_on_collision: bool = True,
                 noisy_ant_initialization: bool = False,
                 max_goal_velocity: float = np.inf,
                 optimize_maze_boarders=True):
        """
        The only different argument is target_log_list which is a list of targets for this envrionment
        instead of a single target

        for more info look at MazeEnv.__init__
        """

        self.target_list = target_loc_list
        self.target_count = len(target_loc_list)

        super().__init__(maze_size=maze_size, maze_map=maze_map, tile_size=tile_size,
                         start_loc=start_loc, target_loc=target_loc_list[0], rewards=rewards,
                         timeout_steps=timeout_steps, show_gui=show_gui, xy_in_obs=xy_in_obs,
                         hit_target_epsilon=hit_target_epsilon, done_on_collision=done_on_collision,
                         noisy_ant_initialization=noisy_ant_initialization, goal_max_velocity=max_goal_velocity,
                         optimize_maze_boarders=optimize_maze_boarders)

    def reset(self, create_video=False, video_path=None, reset_episode_count=False, target_index=None):
        """
        The only new argument is the target index, which determines which target from the targets list
        will be the the target for the next episode. if None, a random target from the list will be chosen
        """
        if target_index is None:
            target_index = np.random.randint(low=0, high=self.target_count)
        if target_index > self.target_count:
            raise Exception("Target index out of bound")

        self.set_target_loc(self.target_list[target_index])

        return super().reset(create_video, video_path, reset_episode_count)

