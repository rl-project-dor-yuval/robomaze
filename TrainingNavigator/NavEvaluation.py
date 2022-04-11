import os

import torch
from stable_baselines3.common.callbacks import BaseCallback
from TrainingNavigator.NavigatorEnv import MultiStartgoalNavigatorEnv
import wandb


class NavEvalCallback(BaseCallback):
    def __init__(self, dir: str, eval_env: MultiStartgoalNavigatorEnv, wandb_run: wandb.run,
                 eval_freq: int = 200, eval_video_freq=-1, verbose=1):
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
        self.verbose = verbose

        wandb_run.define_metric('step', hidden=True)
        wandb_run.define_metric('eval_avg_reward', step_metric='step')
        wandb_run.define_metric('eval_avg_length', step_metric='step')
        wandb_run.define_metric('eval_success_rate', step_metric='step')
        # self.model_save_path = os.path.join(dir, 'best_model')

    def _init_callback(self) -> None:
        super(NavEvalCallback, self)._init_callback()

    def _on_step(self) -> bool:
        super(NavEvalCallback, self)._on_step()
        if self.n_calls % self.eval_freq == 0:
            avg_reward, avg_length, success_rate = self._evaluate_all_workspaces()
            wandb.log({'eval_avg_reward': avg_reward, 'eval_avg_length': avg_length,
                       'eval_success_rate': success_rate, 'step': self.n_calls})

            # TODO : save model
            # TODO : save video



    def _on_training_end(self) -> None:
        super(NavEvalCallback, self)._on_training_end()

    def _on_insert_demo(self) -> None:
        pass # use this?

    def _evaluate_all_workspaces(self):
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

        return avg_reward, avg_length, success_rate

