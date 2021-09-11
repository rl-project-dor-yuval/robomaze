# TODO: explain what this file does

import os
import numpy as np
from dataclasses import dataclass
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from MazeEnv import MazeEnv


class EvalAndSaveCallback(BaseCallback):
    def __init__(self, log_dir: str, eval_env: MazeEnv, eval_freq: int = 200,
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
        super(EvalAndSaveCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
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

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.evals_count += 1
    
            _ = self.eval_env.reset()
            # evaluate the model
            rewards, lengths = evaluate_policy(self.model, self.eval_env, self.eval_episodes,
                                               deterministic=True, return_episode_rewards=True)

            self.step.append(self.n_calls)
            self.mean_reward.append(sum(rewards) / float(len(rewards)))
            self.mean_ep_len.append(sum(lengths) / float(len(lengths)))

            if self.verbose > 0:
                print("{steps} Steps evaluation, avg reward:{reward}, avg episode length: {length}".format(
                    steps=self.n_calls, reward=self.mean_reward[-1], length=self.mean_ep_len[-1]))
            if self.verbose > 1:
                print("rewards:", rewards)
                print("episode lengths:", lengths)

            if self.mean_reward[-1] > self.best_mean_reward:
                self.best_mean_reward = self.mean_reward[-1]
                if self.verbose > 0:
                    print("--Saving new best smodel--")
                self.model.save(self.model_save_path)
                
            if self.eval_video_freq != -1 and self.evals_count % self.eval_video_freq == 0:
                if self.verbose > 0:
                    print("creating video")
                self._create_video()
                
        return True

    def _on_training_end(self) -> None:
        # save results to csv file
        steps_arr = np.expand_dims(np.array(self.step), 1)
        mean_reward_arr = np.expand_dims(np.array(self.mean_reward), 1)
        mean_ep_len_arr = np.expand_dims(np.array(self.mean_ep_len), 1)

        results = np.concatenate((steps_arr, mean_reward_arr, mean_ep_len_arr), axis=1)

        np.savetxt(self.result_save_path, results, delimiter=',')

    def _create_video(self):
        video_path = os.path.join(self.log_dir, '{steps}_steps.avi'.format(steps=self.n_calls))
        obs = self.eval_env.reset(create_video=True, video_path=video_path)
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _ = self.eval_env.step(action)
            if done:
                break


def moving_average(x, kernel_size=7):
    """
     A helper function to compute moving average.
     An array of the same shape is returned with zeros at the first
     kernel_size-1 elements.
     """
    res = np.convolve(x, np.ones(kernel_size), 'valid') / kernel_size
    res = np.concatenate([np.zeros(kernel_size-1), res])
    return res