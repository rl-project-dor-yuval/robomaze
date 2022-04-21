import os
import pathlib
import time
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from TrainingNavigator.NavigatorEnv import MultiStartgoalNavigatorEnv
import wandb


class NavEvalCallback(BaseCallback):
    def __init__(self, dir: str, eval_env: MultiStartgoalNavigatorEnv, wandb_run: wandb.run,
                 eval_freq: int = 5000, eval_video_freq=-1, save_model_freq=20000, eval_workspaces=100,
                 maze_map: np.ndarray = None, verbose=1):
        """
        :param dir: path to the folder where logs and models will be saved
        :param eval_env: separate environment to evaluate the model on
        :param wandb_run: W&B run object
        :param eval_freq: evaluate the model every eval_freq timesteps
        :param eval_video_freq: record videos every eval_video_freq*eval_freq timesteps
        :param save_model_freq: frequency of saving the model
        :param eval_workspaces: on each evaluation, evaluate just on the first eval_workspaces workspaces
        :param maze_map: map of the maze, used to plot trajectories, if None, no plot is made
        :param verbose: verbosity
        """
        super(NavEvalCallback, self).__init__(verbose)

        self.dir = dir
        self.eval_env = eval_env
        self.wandb_run = wandb_run
        self.eval_freq = eval_freq
        self.eval_video_freq = eval_video_freq
        self.save_model_freq = save_model_freq
        self.eval_workspaces = eval_workspaces
        self.maze_map = maze_map
        self.verbose = verbose

        wandb_run.define_metric('step', hidden=True)
        wandb_run.define_metric('eval_avg_reward', step_metric='step')
        wandb_run.define_metric('eval_avg_length', step_metric='step')
        wandb_run.define_metric('eval_success_rate', step_metric='step')
        wandb_run.define_metric('eval_success_rate', step_metric='step')

        wandb_run.define_metric('eval_video', step_metric='step')
        wandb_run.define_metric('grad_norm', step_metric='step')

        self.model_save_path = dir + '/saved_model'
        pathlib.Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        self.video_dir = dir + '/videos'
        pathlib.Path(self.video_dir).mkdir(parents=True, exist_ok=True)

    def _init_callback(self) -> None:
        super(NavEvalCallback, self)._init_callback()
        wandb.watch(self.model.actor, "gradients")

    def _on_step(self) -> bool:
        super(NavEvalCallback, self)._on_step()
        if self.n_calls % self.eval_freq == 0:
            avg_reward, avg_length, success_rate = self._evaluate_all_workspaces()
            self.wandb_run.log({'eval_avg_reward': avg_reward, 'eval_avg_length': avg_length,
                                'eval_success_rate': success_rate, 'step': self.n_calls})

        if self.eval_video_freq > 0 and \
                self.n_calls % (self.eval_video_freq * self.eval_freq) == 0:
            vid_path, walked_trajectory, ws_id = self._record_video()

            log_item = {'eval_video': wandb.Video(vid_path, fps=24), 'step': self.n_calls}
            if self.maze_map is not None:
                log_item['video_trajectory'] = self._get_trajectory_plot(walked_trajectory, ws_id)

            self.wandb_run.log(log_item)

        if self.n_calls % self.save_model_freq == 0:
            if self.verbose > 0:
                print('--- Saving model to {}'.format(self.model_save_path))
            self.model.save(self.model_save_path + '/model_' + str(self.n_calls))

        self._log_actor_critic_grad_norm()

    def _on_training_end(self) -> None:
        super(NavEvalCallback, self)._on_training_end()
        self.model.save(self.model_save_path + '/last_model_' + str(self.n_calls))

    def _log_actor_critic_grad_norm(self) -> None:

        actor_total_norm, critic_total_norm = 0, 0
        actor_params = [p for p in self.model.actor.parameters() if p.grad is not None and p.requires_grad]
        critic_params = [p for p in self.model.critic.parameters() if p.grad is not None and p.requires_grad]

        for p_ac, p_cr in zip(actor_params, critic_params):
            p_ac_norm = p_ac.grad.detach().data.norm(2)
            actor_total_norm += p_ac_norm.item() ** 2

            p_cr_norm = p_ac.grad.detach().data.norm(2)
            critic_total_norm += p_cr_norm.item() ** 2

        actor_total_norm = actor_total_norm ** 0.5
        critic_total_norm = critic_total_norm ** 0.5

        wandb.log({'actor_grad_norm': actor_total_norm, 'step': self.n_calls})
        wandb.log({'critic_grad_norm': critic_total_norm, 'step': self.n_calls})

    def _on_insert_demo(self) -> None:
        pass  # use this?

    def _evaluate_all_workspaces(self):
        t_start = time.time()

        rewards = []
        episodes_length = []
        success_count = 0

        for i in range(self.eval_workspaces):
            obs = self.eval_env.reset(start_goal_pair_idx=i)
            curr_reward = 0
            steps = 0
            while True:
                with torch.no_grad():
                    action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                curr_reward += reward
                steps += 1
                if done:
                    break
            rewards.append(curr_reward)
            episodes_length.append(steps)
            success_count += info['success']

        avg_reward = sum(rewards) / self.eval_workspaces
        avg_length = sum(episodes_length) / self.eval_workspaces
        success_rate = success_count / self.eval_workspaces

        if self.verbose > 0:
            print("All workspaces evaluation done in %.4f secs: " % (time.time() - t_start))

        return avg_reward, avg_length, success_rate

    def _record_video(self) -> Tuple[str, np.ndarray, int]:
        """
        Record a video of the current model
        :return: path to the video, array of the walked trajectory, workspace id
        """
        t_start = time.time()

        video_path = os.path.join(self.video_dir, str(self.n_calls) + '.gif')
        obs = self.eval_env.reset(create_video=True, video_path=video_path)
        walked_traj = []
        while True:
            with torch.no_grad():
                action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, info = self.eval_env.step(action)
            walked_traj.append(obs[:2])
            if done:
                break
        ws_id = info['start_goal_pair_idx']
        self.eval_env.reset()  # to make sure video is saved

        if self.verbose > 0:
            print("Video recorded in %.4f secs: " % (time.time() - t_start))

        return video_path, np.array(walked_traj), ws_id

    def _get_trajectory_plot(self, walked_traj: np.ndarray, ws_id: int) -> plt.Figure:
        """
        returns a plot of the walked trajectory near the RRT planned
         trajectory for that workspace
        """
        walked_traj = walked_traj * 10

        with np.load(self.model.demonstrations_path) as demos:
            planned_traj = demos[str(ws_id)] * 10

        start, goal = planned_traj[0], planned_traj[-1]

        fig, axes = plt.subplots(1, 2)
        for ax in axes:
            ax.imshow(self.maze_map, cmap='gray')
            ax.plot(start[1], start[0], 'go')
            ax.plot(goal[1], goal[0], 'g+')

        axes[0].set_title('Walked trajectory')
        axes[0].plot(walked_traj[:, 1], walked_traj[:, 0], 'ro')
        axes[1].set_title('RRT planned trajectory')
        axes[1].plot(planned_traj[:, 1], planned_traj[:, 0], 'ro')

        return fig
