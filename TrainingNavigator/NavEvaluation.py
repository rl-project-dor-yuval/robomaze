import os
import pathlib
import time

import torch
from stable_baselines3.common.callbacks import BaseCallback
from TrainingNavigator.NavigatorEnv import MultiStartgoalNavigatorEnv
import wandb


class NavEvalCallback(BaseCallback):
    def __init__(self, dir: str, eval_env: MultiStartgoalNavigatorEnv, wandb_run: wandb.run,
                 eval_freq: int = 5000, eval_video_freq=-1, save_model_freq=20000 , verbose=1):
        """
        :param dir: path to the folder where logs and models will be saved
        :param eval_env: separate envrionment to evaluate the model on
        :param eval_freq: evaluate the model every eval_freq timesteps
        :param eval_video_freq: record videos every eval_video_freq*eval_freq timesteps
        :param verbose: verbosity
        """
        super(NavEvalCallback, self).__init__(verbose)

        self.dir = dir
        self.eval_env = eval_env
        self.wandb_run = wandb_run
        self.eval_freq = eval_freq
        self.eval_video_freq = eval_video_freq
        self.save_model_freq = save_model_freq
        self.verbose = verbose

        wandb_run.define_metric('step', hidden=True)
        wandb_run.define_metric('eval_avg_reward', step_metric='step')
        wandb_run.define_metric('eval_avg_length', step_metric='step')
        wandb_run.define_metric('eval_success_rate', step_metric='step')
        wandb_run.define_metric('eval_success_rate', step_metric='step')

        wandb_run.define_metric('eval_video', step_metric='step')

        self.model_save_path = dir + '/saved_model'
        pathlib.Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        self.video_dir = dir + '/videos'
        pathlib.Path(self.video_dir).mkdir(parents=True, exist_ok=True)

    def _init_callback(self) -> None:
        super(NavEvalCallback, self)._init_callback()

    def _on_step(self) -> bool:
        super(NavEvalCallback, self)._on_step()
        if self.n_calls % self.eval_freq == 0:
            avg_reward, avg_length, success_rate = self._evaluate_all_workspaces()
            self.wandb_run.log({'eval_avg_reward': avg_reward, 'eval_avg_length': avg_length,
                       'eval_success_rate': success_rate, 'step': self.n_calls})

        if self.eval_video_freq > 0 and \
                self.n_calls % (self.eval_video_freq * self.eval_freq) == 0:
            vid_path = self._record_video()
            self.wandb_run.log({'eval_video': wandb.Video(vid_path, fps=24), 'step': self.n_calls})

        if self.n_calls % self.save_model_freq == 0:
            if self.verbose > 0:
                print('--- Saving model to {}'.format(self.model_save_path))
            self.model.save(self.model_save_path + '/model_' + str(self.n_calls))

    def _on_training_end(self) -> None:
        super(NavEvalCallback, self)._on_training_end()
        self.model.save(self.model_save_path + '/last_model_' + str(self.n_calls))

    def _on_insert_demo(self) -> None:
        pass  # use this?

    def _evaluate_all_workspaces(self):
        t_start = time.time()

        rewards = []
        episodes_length = []
        success_count = 0

        num_workspaces = self.eval_env.start_goal_pairs_count
        for i in range(num_workspaces):
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

        avg_reward = sum(rewards) / num_workspaces
        avg_length = sum(episodes_length) / num_workspaces
        success_rate = success_count / num_workspaces

        if self.verbose > 0:
            print("All workspaces evaluation done in %.4f secs: " % (time.time() - t_start))

        return avg_reward, avg_length, success_rate

    def _record_video(self) -> str:
        """
        Record a video of the current model
        :return: path to the video
        """
        t_start = time.time()

        video_path = os.path.join(self.video_dir, str(self.n_calls) + '.gif')
        obs = self.eval_env.reset(create_video=True, video_path=video_path)
        while True:
            with torch.no_grad():
                action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = self.eval_env.step(action)
            if done:
                break
        self.eval_env.reset()  # to make sure video is saved

        if self.verbose > 0:
            print("Video recorded in %.4f secs: " % (time.time() - t_start))

        return video_path
