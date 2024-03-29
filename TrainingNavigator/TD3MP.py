import math
import os
from typing import Optional, Union, Type, Tuple, Dict, Any, List

import numpy as np
import torch as th
from stable_baselines3 import TD3
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TrainFreq, RolloutReturn, Schedule, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps, polyak_update
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from stable_baselines3.td3.policies import TD3Policy, Actor
from torch.nn import functional as F
from TrainingNavigator.Utils import trajectory_to_transitions, trajectory_to_transitions_with_heading


class TD3MP(TD3):
    def __init__(
            self,
            policy: Union[str, Type[TD3Policy]],
            env: SubprocVecEnv,
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
            policy_delay: int = 2,
            target_policy_noise: float = 0.2,
            target_noise_clip: float = 0.5,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            demonstrations_path: os.path = None,
            demo_on_fail_prob: float = 0.5,
            demo_prob_decay: float = 1,
            use_demo_epsilon_offset: bool = True,
            grad_clip_norm_actor: float = None,
            grad_clip_norm_critic: float = None,
    ):
        super(TD3MP, self).__init__(
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
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            optimize_memory_usage=optimize_memory_usage,
        )

        self.demonstrations_path = demonstrations_path
        self.demo_on_fail_prob = demo_on_fail_prob
        self.grad_clip_norm_actor = grad_clip_norm_actor
        self.grad_clip_norm_critic = grad_clip_norm_critic
        self.demo_prob_decay = demo_prob_decay
        self.use_demo_epsilon_offset = use_demo_epsilon_offset

        self.n_demos_inserted = 0

        # keep for comfort:
        self.epsilon_to_goal = env.get_attr('epsilon_to_hit_subgoal', 0)[0]
        self.normalize_obs = self.env.get_attr('normalize_observations', 0)[0]
        self.control_heading_at_subgoal = self.env.get_attr('control_heading_at_subgoal', 0)[0]
        self.maze_size = self.env.get_attr('maze_env', 0)[0].maze_size

        # keep a small buffer for demonstrations, it is used in _insert_demo_to_replay_buffer
        num_envs = self.n_envs
        self.obs_buff = np.zeros((num_envs,) + self.observation_space.shape)
        self.new_obs_buff = np.zeros((num_envs,) + self.observation_space.shape)
        self.action_buff = np.zeros((num_envs, self.action_space.shape[0]))
        self.reward_buff = np.zeros(num_envs)
        self.done_buff = np.zeros(num_envs)
        self.info_buff = [{}] * num_envs
        self.curr_buff_idx = 0

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        override td3 method but copy of the same implementation just added gradient logging and clipping
        """

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_clip_norm_critic is not None:
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm_critic)
            self._log_grad_norm(self.critic, "train/critic_grad_norm")
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations,
                                                     self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                if self.grad_clip_norm_actor is not None:
                    th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm_actor)
                self._log_grad_norm(self.actor, "train/actor_grad_norm")
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

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
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes,
                                     continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

                    info = infos[idx]
                    if info['hit_maze'] or info['fell'] or info['TimeLimit.truncated']:
                        # insert demonstration to replay buffer with probability self.demo_on_fail_prob
                        if np.random.rand() < self.demo_on_fail_prob:
                            if self.verbose > 0:
                                print("Failed Episode. Inserting demonstration to replay buffer.")

                            # It's inefficient to load the whole demo buffer every time, then create
                            # transitions from it, but it's still negligible compared to the time it takes
                            # to simulate one episode. Maybe one day we can optimize this.
                            with np.load(self.demonstrations_path) as demos:
                                demo_traj = demos[str(info['workspace_idx'])]
                            self._insert_demo_to_replay_buffer(replay_buffer, demo_traj,
                                                               info['workspace_idx'])

                            self.n_demos_inserted += 1
                            self.demo_on_fail_prob *= self.demo_prob_decay

                            self.logger.record("demonstrations/n_demos_inserted", self.n_demos_inserted)
                            self.logger.record("demonstrations/demo_prob", self.demo_on_fail_prob)

                        elif self.verbose > 0:
                            print("Failed Episode, but not inserting demonstration to replay buffer.")

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _insert_demo_to_replay_buffer(self, replay_buffer, demo_traj, demo_traj_idx):
        # because replay buffer stores transitions in arrays of size buffer_size*num_envs,
        # we need to collect transitions into arrays of num_envs to insert them to the replay buffer,
        # therefore we insert just every num_envs'th transition from the demo trajectory
        # until we collect that number of transitions, we save them in temporary buffers.

        rewards_config = self.env.get_attr('rewards_config', 0)[0]
        observations, actions, rewards, next_observations, dones = \
            trajectory_to_transitions_with_heading(demo_traj, rewards_config, self.epsilon_to_goal) \
            if self.control_heading_at_subgoal \
            else trajectory_to_transitions(demo_traj, rewards_config, self.epsilon_to_goal)

        for i in range(len(observations)):
            self.obs_buff[self.curr_buff_idx] = self.normalize_obs_if_needed(observations[i])
            self.new_obs_buff[self.curr_buff_idx] = self.normalize_obs_if_needed(next_observations[i])

            # important - rescale action!
            actions[i] = self.policy.scale_action(actions[i])
            self.action_buff[self.curr_buff_idx] = np.clip(actions[i], -1, 1)

            self.done_buff[self.curr_buff_idx] = dones[i]
            self.reward_buff[self.curr_buff_idx] = rewards[i]

            self.info_buff[self.curr_buff_idx] = {'hit_maze': False, 'fell': False, 'stepper_timeout': False,
                                                  'navigator_timeout': False, 'workspace_idx': demo_traj_idx,
                                                  'TimeLimit.truncated': False}
            if i == len(observations) - 1:
                self.info_buff[self.curr_buff_idx]['success'] = True
            else:
                self.info_buff[self.curr_buff_idx]['success'] = False

            if self.curr_buff_idx + 1 != replay_buffer.n_envs:
                # dont insert yet
                self.curr_buff_idx += 1
                continue

            # collected enough transitions, insert them to the replay buffer
            replay_buffer.add(
                self.obs_buff,
                self.new_obs_buff,
                self.action_buff,
                self.reward_buff,
                self.done_buff,
                self.info_buff,
            )
            self.curr_buff_idx = 0

    def _excluded_save_params(self) -> List[str]:
        return super(TD3MP, self)._excluded_save_params() + ["demonstrations"]

    def _log_grad_norm(self, policy, log_name) -> None:
        total_norm = 0
        params = [p for p in policy.parameters() if p.grad is not None and p.requires_grad]

        for p in params:
            p_norm = p.grad.detach().data.norm(2)
            total_norm += p_norm.item() ** 2

        total_norm = total_norm ** 0.5

        self.logger.record(log_name, total_norm)

    # noinspection DuplicatedCode
    def normalize_obs_if_needed(self, obs):
        """
        almost copy of implementation in NavigatorEnv Because we dont want to invoke methods of object in other proccesses
        """
        norm_obs = obs.copy()
        if self.normalize_obs:
            # normalize x and y of robot and goal:
            maze_size_x, maze_size_y = self.maze_size
            max_xy = np.array([maze_size_x, maze_size_y])
            norm_obs[0:2] = 2 * (norm_obs[0:2] / max_xy) - 1
            norm_obs[3:5] = 2 * (norm_obs[3:5] / max_xy) - 1

            # normalize rotation:
            norm_obs[2] = norm_obs[2] / np.pi

        return norm_obs

    def normalize_tarj_if_needed(self, demo_traj):
        """ delete this?"""
        if self.normalize_obs:
            demo_traj[:, 0] /= self.maze_size[0]
            demo_traj[:, 1] /= self.maze_size[1]
            demo_traj *= 2
            demo_traj -= 1
        return demo_traj


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

        dx, dy, rotation = out[:, 0], out[:, 1], out[:, 2]
        r, theta = th.sqrt(dx ** 2 + dy ** 2), th.atan2(dy, dx)

        # scale the action. all sizes (except for low, high) are torch tensors to assure differentiation
        low, high = self.action_space.low, self.action_space.high
        # r_scaled = 2 * (r - low[0]) / (high[0] - low[0]) - 1
        # r_scaled = r_scaled.clip(-1, 1)
        r_scaled = th.tanh(r)
        theta_scaled = 2 * (theta - low[1]) / (high[1] - low[1]) - 1
        # theta_scaled = theta_scaled.clip(-1, 1)
        rotation_scaled = 2 * (rotation - low[2]) / (high[2] - low[2]) - 1

        return th.stack([r_scaled, theta_scaled, rotation_scaled], dim=1)


class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)


def replay_buffer_debug(replay_buffer):
    pass
