import os
import abc
import time
from typing import Tuple, List
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import imageio
import glob
import sys
import wandb

sys.path.append('..')
from MazeEnv.MazeEnv import MazeEnv
from MazeEnv.MultWorkspaceMazeEnv import MultiWorkspaceMazeEnv


class BaseEvalAndSaveCallback(BaseCallback):
    def __init__(self, log_dir: str, eval_env: MazeEnv, wb_run: wandb.run, eval_freq: int = 200,
                 eval_episodes: int = 5, eval_video_freq=-1, verbose=1):
        """
        :param log_dir: The directory in which to put the results
        :param eval_env: MazeEnv to use for evaluation, should not be the original
         environment since evaluation may happen in a middle of an episode
        :param eval_freq: Evaluation happens every eval_freq steps
        :param eval_episodes: How mach episodes will be ran for evaluation (results is averaged over them)
        :param eval_video_freq: create video every eval_video_freq evaluations. -1 will not create any video.
        :param verbose: verbose
        """
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.wb_run = wb_run
        self.log_dir = log_dir
        self.eval_env = eval_env
        self.eval_video_freq = eval_video_freq

        self.evals_count = 0

        self.model_save_path = os.path.join(log_dir, 'best_model')
        self.result_save_path = os.path.join(log_dir, 'eval_results.csv')

        self.best_mean_reward = -np.inf

        self.step = []
        self.mean_reward = []
        self.mean_ep_len = []
        self.success_rate = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.evals_count += 1

            # evaluate the model
            start_time = time.time()
            rewards, lengths, success_rate, mean_terminal_vel = self.get_policy_evaluation()
            eval_time = time.time() - start_time
            if self.verbose >= 1:
                print("Evaluation took {:.2f}s".format(eval_time))

            self.step.append(self.n_calls)
            self.mean_reward.append(sum(rewards) / float(len(rewards)))
            self.mean_ep_len.append(sum(lengths) / float(len(lengths)))
            self.success_rate.append(success_rate)

            if self.verbose > 0:
                print(f"{self.n_calls:.2f} Steps evaluation, avg reward:{self.mean_reward[-1]:.2f},"
                      f" avg episode length: {self.mean_ep_len[-1]:.2f}, success rate: {self.success_rate[-1]:.2f}")
            if self.verbose > 1:
                print("rewards:", rewards)
                print("episode lengths:", lengths)

            if self.mean_reward[-1] > self.best_mean_reward:
                self.best_mean_reward = self.mean_reward[-1]
                if self.verbose > 0:
                    print("--Saving new best model--")
                self.model.save(self.model_save_path)

            # save current model anyway, don't be cheap :)
            self.model.save(os.path.join(self.log_dir, 'model_' + str(self.n_calls)))

            if self.eval_video_freq != -1 and self.evals_count % self.eval_video_freq == 0:
                if self.verbose > 0:
                    print("creating video")
                video_path = self._create_video()
                vid_log_item = log_item = {'eval_video': wandb.Video(video_path, fps=40), 'step': self.n_calls}
                self.wb_run.log(vid_log_item)

                # The video that is being created here is different from the
                # case that made the `rewards` above

            self.wb_run.log({'success_rate': success_rate, 'mean_terminal_velocity': mean_terminal_vel})

        return True

    def _on_training_end(self) -> None:
        # reset env to save last video:
        _ = self.eval_env.reset()

        # save results to csv file
        steps_arr = np.expand_dims(np.array(self.step), 1)
        mean_reward_arr = np.expand_dims(np.array(self.mean_reward), 1)
        mean_ep_len_arr = np.expand_dims(np.array(self.mean_ep_len), 1)
        success_rate_arr = np.expand_dims(np.array(self.success_rate), 1)

        results = np.concatenate((steps_arr, mean_reward_arr, mean_ep_len_arr, success_rate_arr), axis=1)

        np.savetxt(self.result_save_path, results, delimiter=',')

    def _create_video(self):
        video_path = os.path.join(self.log_dir, '{steps}_steps.gif'.format(steps=self.n_calls))
        obs = self.eval_env.reset(create_video=True, video_path=video_path)
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _ = self.eval_env.step(action)
            if done:
                break

        return video_path

    @abc.abstractmethod
    def get_policy_evaluation(self) -> Tuple[List[float], List[int], float, float]:
        """
        :return: tuple of reward for each episode and length of each episode in the evaluation
        """
        pass


class EvalAndSaveCallback(BaseEvalAndSaveCallback):
    def get_policy_evaluation(self) -> Tuple[List[float], List[int], float, float]:
        rewards = []
        episodes_length = []
        terminal_velocities = []
        success_rate_cnt = 0

        # evaluate on all targets
        for _ in range(self.eval_episodes):
            # play episode:
            obs = self.eval_env.reset()
            step_count = 0
            total_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                step_count += 1
                total_reward += reward

            # save results:
            rewards.append(total_reward)
            episodes_length.append(step_count)
            success_rate_cnt += float(info['success'])
            terminal_velocities.append(np.sqrt(obs[3] ** 2 + obs[4] ** 2))

        success_rate = success_rate_cnt / self.eval_episodes
        mean_terminal_velocity = sum(terminal_velocities) / self.eval_episodes
        return rewards, episodes_length, success_rate, mean_terminal_velocity


class MultiWorkspaceEvalAndSaveCallback(BaseEvalAndSaveCallback):
    """
    Only works with MultiWorkspaceMazeEnv.
    Each evaluation is for all the targets in the environment and result is averaged
    """
    eval_env: MultiWorkspaceMazeEnv

    def __init__(self, log_dir: str, eval_env: MultiWorkspaceMazeEnv, wb_run: wandb.run, eval_freq: int = 200,
                 eval_video_freq=-1, verbose=1):
        eval_episodes = eval_env.workspace_count
        super().__init__(log_dir, eval_env, wb_run, eval_freq, eval_episodes,
                         eval_video_freq, verbose)

        self.per_target_success = np.zeros(eval_episodes)

    def get_policy_evaluation(self) -> Tuple[List[float], List[int], float, float]:
        rewards = []
        episodes_length = []
        terminal_velocities = []
        success_rate_cnt = 0

        # evaluate on all targets
        for i in range(self.eval_episodes):
            # play episode:
            obs = self.eval_env.reset(workspace_index=i)
            step_count = 0
            total_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                step_count += 1
                total_reward += reward

            # save results:
            rewards.append(total_reward)
            episodes_length.append(step_count)
            success_rate_cnt += float(info['success'])
            self.per_target_success[i] += float(info['success'])
            terminal_velocities.append(np.sqrt(obs[3] ** 2 + obs[4] ** 2))

        success_rate = success_rate_cnt / self.eval_episodes
        mean_terminal_velocity = sum(terminal_velocities) / self.eval_episodes
        return rewards, episodes_length, success_rate, mean_terminal_velocity

    def _on_training_end(self) -> None:
        super()._on_training_end()

        # save per target success rate as well :
        per_target_success_rate = self.per_target_success / self.evals_count
        np.savetxt(os.path.join(self.log_dir, 'per_target_success_rate.csv'),
                   per_target_success_rate)




def moving_average(x, kernel_size=7):
    """
     A helper function to compute moving average.
     An array of the same shape is returned with zeros at the first
     kernel_size-1 elements.
     """
    # in order to fit plot when low number of samples.
    if kernel_size > len(x):
        kernel_size = len(x)
    res = np.convolve(x, np.ones(kernel_size), 'valid') / kernel_size
    res = np.concatenate([x[0] * np.ones(kernel_size - 1), res])
    return res


def plot_train_eval_results(log_dir, n_eval_episodes):
    """
    plots 4 graphs:
    - Evaluation success rate
    - Evaluation avg Reward
    - Evaluation avg episode length
    - Evaluation avg episode length moving average

    :param log_dir: the log dir that passed to the model. must contain:
        results.monitor.csv and eval_results.csv
    :param n_eval_episodes: the num of episodes that the evaluation reward was averaged on
    :return: None
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    eval_results_df = pd.read_csv(os.path.join(log_dir, "eval_results.csv"),
                                  names=["Step", "Avg Reward", "Avg Episode Length", "Success Rate"],
                                  index_col=0)

    eval_results_df.plot(y="Avg Reward", ax=axes[0, 0], legend=None)
    axes[0, 0].set_title("Evaluation Reward (avg over {eval_episodes} Episodes)".format(eval_episodes=n_eval_episodes))

    eval_results_df.plot(y="Success Rate", ax=axes[0, 1], legend=None)
    axes[0, 1].set_title("Evaluation Success Rate Over {eval_episodes} Episodes)".format(eval_episodes=n_eval_episodes))

    eval_results_df.plot(y="Avg Episode Length", ax=axes[1, 0], legend=None)
    axes[1, 0].set_title("Evaluation Avg Episode Length")

    steps = eval_results_df.index.to_numpy()
    eval_length = eval_results_df["Avg Episode Length"].to_numpy()
    eval_length_moving_avg = moving_average(eval_length, kernel_size=20)
    axes[1, 1].plot(steps, eval_length_moving_avg)
    axes[1, 1].set_title("Evaluation Avg Episode Length Moving Average")
    axes[1, 1].set_xlabel("Step")


def record_model(model, env, video_path):
    """
    create video of a model doing one episode and save it
    :param model: the model to record playing
    :param env: the env to record on
    :param video_path: the path to save the video in including file name
    :return: the reward of the episode
    """
    # evaluate the best model
    episode_reward = 0
    obs = env.reset(create_video=True, video_path=video_path)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            env.reset()
            return episode_reward


def create_gifs_from_avi(log_directory_path):
    in_files_path = glob.glob(log_directory_path + "/*.avi")
    out_files_path = [os.path.splitext(in_pth)[0] + ".gif" for in_pth in in_files_path]

    for in_path, out_path in zip(in_files_path, out_files_path):
        reader = imageio.get_reader(in_path)
        fps = reader.get_meta_data()['fps']
        writer = imageio.get_writer(out_path, fps=fps)

        for i, im in enumerate(reader):
            writer.append_data(im)

        writer.close()

