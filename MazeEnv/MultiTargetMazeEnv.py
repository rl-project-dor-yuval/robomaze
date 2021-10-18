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
                 xy_in_obs: bool = True):
        """
        The only different argument is target_log_list which is a list of targets for this envrionment
        instead of a single target

        for more info look at MazeEnv.__init__
        """

        self.target_list = target_loc_list
        self.target_count = len(target_loc_list)

        super().__init__(maze_size, maze_map, tile_size, start_loc, target_loc_list[0],
                         rewards, timeout_steps, show_gui, xy_in_obs)

    def reset(self, create_video=False, video_path=None, reset_episode_count=False, target_index=None):
        """
        The only new argument is the target index, which determines which target from the targets list
        will be the the target for the next episode. if None, a random target from the list will be chosen
        """
        if target_index is None:
            target_index = np.random.randint(low=0, high=self.target_count)
        if target_index > self.target_count:
            raise Exception("Target index out of bound")

        new_target_xy = self.target_list[target_index]
        self._target_loc = (new_target_xy[0], new_target_xy[1], 0)

        # move target sphere for visual and collision detection:
        _, target_uid, _ = self._maze.get_maze_objects_uids()
        _, old_orientation = self._pclient.getBasePositionAndOrientation(target_uid)
        self._pclient.resetBasePositionAndOrientation(target_uid, self._target_loc, old_orientation)

        return super().reset(create_video, video_path, reset_episode_count)

