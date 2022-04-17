import os
from typing import Optional, Union, Type, Tuple, Dict, Any, List
import torch as th
import numpy as np
from NavigatorEnv import MultiStartgoalNavigatorEnv
import math

from stable_baselines3 import DDPG
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import TrainFreq, RolloutReturn, GymEnv, Schedule
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.td3.policies import TD3Policy, Actor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DDPGMP(DDPG):
    def __init__(
            self,
            policy: Union[str, Type[TD3Policy]],
            env: MultiStartgoalNavigatorEnv,
            learning_rate: Union[float, Schedule] = 1e-3,
            buffer_size: int = 1000000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 100,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
            gradient_steps: int = -1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[ReplayBuffer] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            demonstrations_path: os.path = None,
            demo_on_fail_prob: float = 0.5,
    ):
        super(DDPGMP, self).__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
        )

        # self.demonstrations = np.load(demonstrations_path)
        # if verbose > 0:
        #     print(f"Debug: loaded {len(self.demonstrations)} different demonstrations")
        self.demonstrations_path = demonstrations_path
        self.demo_on_fail_prob = demo_on_fail_prob

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBuffer,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Same as original implementation in OffPolicyAlgorithm but with the addition
        of inserting demonstrations to the replay buffer in failure
        """
        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)

                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if done:
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

                info = infos[0]
                if info['hit_maze'] or info['fell'] or info['navigator_timeout']:
                    # insert demonstration to replay buffer with probability self.demo_on_fail_prob
                    if np.random.rand() < self.demo_on_fail_prob:
                        if self.verbose > 0:
                            print("Failed Episode. Inserting demonstration to replay buffer.")

                        with np.load(self.demonstrations_path) as demos:
                            demo_traj = demos[str(info['start_goal_pair_idx'])]
                        self._insert_demo_to_replay_buffer(replay_buffer, demo_traj,
                                                           info['start_goal_pair_idx'])
                    elif self.verbose > 0:
                        print("Failed Episode, but not inserting demonstration to replay buffer.")

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)

    def _insert_demo_to_replay_buffer(self, replay_buffer, demo_traj, demo_traj_idx):
        for i in range(len(demo_traj) - 1):
            obs = np.concatenate([demo_traj[i], demo_traj[-1]])
            new_obs = np.concatenate([demo_traj[i + 1], demo_traj[-1]])
            # recall : Observation -> [ Agent_x, Agent_y, Target_x, Target_y]
            action = self._compute_fake_action(obs, new_obs)
            # important - rescale action!
            action = self.policy.scale_action(action)
            action = np.clip(action, -1, 1)
            # TODO - consider training on normalized action, maybe tanh

            if i == len(demo_traj) - 2:  # last transition:
                reward = self.env.envs[0].rewards_config.target_arrival
                done = True
                info = {'hit_maze': False, 'fell': False,  'stepper_timeout': False,
                        'navigator_timeout': False, 'start_goal_pair_idx': demo_traj_idx,
                        'success': True}
            else:
                reward = self.env.envs[0].rewards_config.idle
                done = False
                info = {'hit_maze': False, 'fell': False, 'stepper_timeout': False,
                        'navigator_timeout': False, 'start_goal_pair_idx': demo_traj_idx,
                        'success': False}

            replay_buffer.add(
                obs,
                new_obs,
                action,
                reward,
                done,
                [info],
            )

    @staticmethod
    def _compute_fake_action(obs, new_obs):
        dx, dy = new_obs[0] - obs[0], new_obs[1] - obs[1]
        r, theta = math.sqrt(dx ** 2 + dy ** 2), math.atan2(dy, dx)
        return np.array([r, theta])

    def _excluded_save_params(self) -> List[str]:
        return super(DDPGMP, self)._excluded_save_params() + ["demonstrations"]


class CustomActor(Actor):
    """
    Custom Actor network (policy) for TD3. applying a transformation to the output of the network to r,theta.
    """

    def __init__(self, *args, **kwargs):
        super(CustomActor, self).__init__(*args, **kwargs)
        # Define custom network with r, theta output
        # WARNING: it must end with a tanh activation to squash the output
        # self.mu = th.nn.Sequential(...)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        # Actor outputs learned delta x, delta y
        out = self.mu(features)
        # transforming to r,theta

        dx, dy = out[:, 0], out[:, 1]
        r, theta = th.sqrt(dx ** 2 + dy ** 2), th.atan2(dy, dx)

        # # scale the action. all sizes (except for low, high) are torch tensors to assure differentiation
        low, high = self.action_space.low, self.action_space.high
        # r_scaled = 2 * (r - low[0]) / (high[0] - low[0]) - 1
        # r_scaled.clip(-1, 1)
        # theta_scaled = 2 * (theta - low[1]) / (high[1] - low[1]) - 1

        r_scaled = r.clip(low[0], high[0])
        theta_scaled = theta.clip(low[1], high[1])

        return th.stack([r_scaled, theta_scaled], dim=1)


class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)


def replay_buffer_debug(replay_buffer):
    import pandas as pd

