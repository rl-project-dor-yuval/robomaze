import numpy as np

from MazeEnv.MazeEnv import MazeEnv
from MazeEnv.EnvAttributes import MazeSize, Rewards, Workspace


class MultiWorkspaceMazeEnv(MazeEnv):
    def __init__(self,
                 maze_size=MazeSize.MEDIUM,
                 maze_map: np.ndarray = None,
                 tile_size=0.1,
                 workspace_list=(Workspace()),
                 rewards: Rewards = Rewards(),
                 timeout_steps: int = 0,
                 show_gui: bool = False,
                 xy_in_obs: bool = True,
                 hit_target_epsilon: float = 0.8,
                 target_heading_epsilon: float = np.pi,
                 done_on_collision: bool = True,
                 done_on_goal_reached: bool = True,
                 success_steps_before_done=1,
                 noisy_robot_initialization: bool = False,
                 max_goal_velocity: float = np.inf,
                 optimize_maze_boarders=True,
                 sticky_actions=8,
                 robot_type: str = 'Ant'):
        """
        A MAzeEnv with multiple workspaces. A random workspace from workspace_list will be chosen for each
        episode unless one is provided in the reset() method.
        """
        self.workspace_list = workspace_list
        self.workspace_count = len(workspace_list)

        super().__init__(maze_size=maze_size,
                         maze_map=maze_map,
                         tile_size=tile_size,
                         workspace=workspace_list[0],
                         rewards=rewards,
                         timeout_steps=timeout_steps,
                         show_gui=show_gui,
                         xy_in_obs=xy_in_obs,
                         hit_target_epsilon=hit_target_epsilon,
                         target_heading_epsilon=target_heading_epsilon,
                         done_on_collision=done_on_collision,
                         done_on_goal_reached=done_on_goal_reached,
                         success_steps_before_done=success_steps_before_done,
                         noisy_robot_initialization=noisy_robot_initialization,
                         goal_max_velocity=max_goal_velocity,
                         optimize_maze_boarders=optimize_maze_boarders,
                         sticky_actions=sticky_actions,
                         robot_type=robot_type)

    def reset(self, create_video=False, video_path=None, reset_episode_count=False, workspace_index=None):
        """
        The only new argument is the workspace index, which determines which workspace from the workspaces list
        will be the workspace for the next episode. if None, a random workspace from the list will be chosen.
        """
        if workspace_index is None:
            workspace_index = np.random.randint(low=0, high=self.workspace_count)
        if workspace_index > self.workspace_count:
            raise Exception("Workspace index out of bound")

        self.set_workspace(self.workspace_list[workspace_index])

        return super().reset(create_video, video_path, reset_episode_count)

