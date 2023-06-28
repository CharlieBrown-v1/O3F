import os
import warnings
from typing import Any, Dict, Optional, Type, Union, Tuple

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm, HybridOnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, HybridPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
            self,
            policy: Union[str, Union[Type[ActorCriticPolicy], Type[HybridPolicy]]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):

        super(PPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
                batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                    buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))

        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "PPO",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            save_interval: Optional[int] = None,
            save_path: Optional[str] = None,
            save_count: int = 0,
    ) -> "OnPolicyAlgorithm":

        return super(PPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            save_interval=save_interval,
            save_path=save_path,
            save_count=save_count,
        )


import time
from stable_baselines3.common.utils import safe_mean
from collections import deque


class HybridPPO(HybridOnPolicyAlgorithm):
    def __init__(
            self,
            policy: Union[str, Union[Type[ActorCriticPolicy], Type[HybridPolicy]]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            is_two_stage_env: bool = False,
            upper_counting_mode: bool = False,
    ):

        super(HybridPPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
            is_two_stage_env=is_two_stage_env,
            upper_counting_mode=upper_counting_mode,
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
                batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                    buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        # DIY
        self.sb3_info_buffer = deque(maxlen=100)

        self.removal_success_buffer = None
        self.global_success_buffer = None

        self.upper_info_buffer = None

        if _init_setup_model:
            self._setup_model()

        if is_two_stage_env:
            self.removal_success_buffer = deque(maxlen=100)
            self.global_success_buffer = deque(maxlen=100)

        if upper_counting_mode:
            self.upper_info_buffer = deque(maxlen=100)

        self.pretrain_epoch = None

    def _setup_model(self) -> None:
        super(HybridPPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self, prefix=None):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            if not continue_training:
                break
        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())
        # Logs
        prefix = f'{prefix}' if prefix is not None else ''

        self.logger.record(f"{prefix}train/entropy_loss", np.mean(entropy_losses))
        self.logger.record(f"{prefix}train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record(f"{prefix}train/value_loss", np.mean(value_losses))
        self.logger.record(f"{prefix}train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record(f"{prefix}train/clip_fraction", np.mean(clip_fractions))
        self.logger.record(f"{prefix}train/loss", loss.item())
        self.logger.record(f"{prefix}train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(f"{prefix}train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record(f"{prefix}train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record(f"{prefix}train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record(f"{prefix}train/clip_range_vf", clip_range_vf)

    def value_function_pretrain(
            self,
            prefix=None,
    ) -> float:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        entropy_losses = []
        value_losses = []
        n_epoch = -1
        for n_epoch in range(self.n_epochs):
            critic_loss_list = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                critic_loss_list.append(value_loss.item())
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

        self.pretrain_epoch += n_epoch + 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())
        # Logs
        prefix = f'{prefix}' if prefix is not None else ''

        self.logger.record(f"{prefix}pretrain/n_epoch", self.pretrain_epoch)
        self.logger.record(f"{prefix}pretrain/entropy_loss", np.mean(entropy_losses))
        self.logger.record(f"{prefix}pretrain/value_loss", np.mean(value_losses))
        self.logger.record(f"{prefix}pretrain/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(f"{prefix}pretrain/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record(f"{prefix}pretrain/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record(f"{prefix}pretrain/clip_range_vf", clip_range_vf)

        return np.mean(value_losses).item()

    def learn_one_step(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "OnPolicyAlgorithm",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            save_interval: Optional[int] = None,
            save_path: Optional[str] = None,
            success_rate_save_path: Optional[str] = None,
            accumulated_save_count: int = 0,
            accumulated_time_elapsed: float = 0.0,
            accumulated_iteration: int = 0,
            accumulated_total_timesteps: int = 0,
            prefix: str = None,
            fine_tuning_flag: bool = False,
            fine_tuning_kwargs: dict = None,
    ) -> "HybridPPO":
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        prefix = f'{prefix} ' if prefix is not None else ''
        success_rate_arr = np.array([0])
        timesteps_arr = np.array([0])

        if fine_tuning_flag:
            stop_criteria_cnt = fine_tuning_kwargs['count']
            self.pretrain_epoch = 0
            critic_loss_list = []
            prev_loss_mean = None
            curr_loss_mean = None
            pretrain_finish = False
            while not pretrain_finish:
                continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                          n_rollout_steps=self.n_steps)
                if continue_training is False:
                    break

                critic_loss = self.value_function_pretrain(prefix=prefix)
                critic_loss_list.append(critic_loss)
                pretrain_iteration = self.pretrain_epoch // self.n_epochs

                if save_interval is not None and pretrain_iteration % save_interval == 0:
                    self.save(save_path + "_pretrain_" + str(pretrain_iteration))
                    self.logger.record(f"{prefix}pretrain/Save Model", pretrain_iteration)

                if pretrain_iteration % stop_criteria_cnt == 0:
                    if prev_loss_mean is None:
                        prev_loss_mean = np.inf
                    else:
                        prev_loss_mean = curr_loss_mean
                    curr_loss_mean = np.mean(critic_loss_list[-stop_criteria_cnt:])
                    self.logger.record(f"{prefix}pretrain/loss_mean", curr_loss_mean)
                    if curr_loss_mean >= prev_loss_mean:
                        print(
                            f'Stop pretraining as critic loss no longer drops({prev_loss_mean:.2f} -> {curr_loss_mean:.2f})')
                        pretrain_finish = True

                self.logger.dump(step=pretrain_iteration)

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            accumulated_iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and accumulated_iteration % log_interval == 0:
                fps = int(accumulated_total_timesteps + self.num_timesteps / (
                        accumulated_time_elapsed + time.time() - self.start_time))
                self.logger.record(f"{prefix}time/iterations", accumulated_iteration, exclude="tensorboard")

                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(f"{prefix}rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record(f"{prefix}rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

                    if self.is_two_stage_env:
                        if len(self.removal_success_buffer) > 0:
                            self.logger.record(f"{prefix}rollout/stage_1 success_rate",
                                               safe_mean(self.removal_success_buffer))
                        if len(self.global_success_buffer) > 0:
                            self.logger.record(f"{prefix}rollout/stage_2 success_rate",
                                               safe_mean(self.global_success_buffer))

                    if self.upper_counting_mode:
                        if len(self.upper_info_buffer) > 0:
                            is_good_goal_arr = np.array([upper_info['is_good_goal']
                                                         for upper_info in self.upper_info_buffer])
                            is_obstacle_chosen_arr = np.array([upper_info['is_obstacle_chosen']
                                                               for upper_info in self.upper_info_buffer])
                            try:
                                good_goal_rate = is_good_goal_arr.sum() / is_obstacle_chosen_arr.sum()
                            except ZeroDivisionError:
                                good_goal_rate = 1
                            self.logger.record(f"{prefix}rollout/good goal rate", good_goal_rate)

                if len(self.ep_success_buffer) > 0:
                    success_rate_mean = safe_mean(self.ep_success_buffer)
                    self.logger.record(f"{prefix}rollout/success_rate", success_rate_mean)
                    success_rate_arr = np.r_[success_rate_arr, success_rate_mean]
                    timesteps_arr = np.r_[timesteps_arr, accumulated_total_timesteps + self.num_timesteps]

                if len(self.sb3_info_buffer) > 0:
                    key_list = list(self.sb3_info_buffer[0].keys())
                    for key in key_list:
                        self.logger.record(f"{prefix}sb3_info/{key}",
                                           safe_mean([ep_info[key] for ep_info in self.sb3_info_buffer]))

                self.logger.record(f"{prefix}time/fps", fps)
                self.logger.record(f"{prefix}time/time_elapsed",
                                   int(accumulated_time_elapsed + time.time() - self.start_time),
                                   exclude="tensorboard")
                self.logger.record(f"{prefix}time/total_timesteps", accumulated_total_timesteps + self.num_timesteps,
                                   exclude="tensorboard")
                self.logger.dump(step=accumulated_total_timesteps + self.num_timesteps)

            # DIY
            if save_interval is not None and accumulated_iteration % save_interval == 0:
                assert save_path is not None
                accumulated_save_count += 1
                self.save(save_path + "_" + str(accumulated_save_count))
                self.logger.record(f"{prefix}Save Model", accumulated_save_count)
                self.logger.record(f"{prefix}time/iterations", accumulated_iteration)
                self.logger.record(f"{prefix}time/total_timesteps", accumulated_total_timesteps + self.num_timesteps)
                self.logger.dump(step=accumulated_total_timesteps + self.num_timesteps)

            self.train(prefix=prefix)

        if success_rate_save_path is not None:
            self.save_success_rate(success_rate_save_path=success_rate_save_path,
                                   success_rate_arr=success_rate_arr,
                                   timesteps_arr=timesteps_arr,
                                   )

        callback.on_training_end()

        return self

    def save_success_rate(self,
                          success_rate_save_path: Optional[str] = None,
                          success_rate_arr: np.ndarray = None,
                          timesteps_arr: np.ndarray = None,
                          ) -> None:
        assert success_rate_save_path is not None and success_rate_arr is not None and timesteps_arr is not None
        np.save(os.path.join(success_rate_save_path, f'seed:{self.seed}_timesteps'), timesteps_arr)
        np.save(os.path.join(success_rate_save_path, f'seed:{self.seed}_success_rate'), success_rate_arr)

    def setup_learn(self, total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path,
                    reset_num_timesteps, tb_log_name):
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )
        return total_timesteps, callback

    def _update_info_buffer(self, infos, dones) -> None:
        if dones is None:
            dones = np.array([False] * len(infos))

        for idx, info in enumerate(infos):
            sb3_info = info.get("sb3_info")
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")

            if sb3_info is not None and dones[idx]:
                self.sb3_info_buffer.extend([sb3_info])
            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)

    def _two_stage_env_update_info_buffer(self, infos, dones):
        assert self.removal_success_buffer is not None and self.global_success_buffer is not None
        assert dones is not None

        for idx, info in enumerate(infos):
            maybe_ep_info = info.get("episode")
            maybe_is_success = info.get("is_success")
            maybe_removal_done = info.get('removal_done') or info.get('TimeLimit.truncated', False)
            maybe_removal_success = info.get('removal_success')
            maybe_global_done = info.get('global_done') or (
                    not info.get('removal_done') and info.get('TimeLimit.truncated', False))
            maybe_global_success = info.get('global_success')

            if maybe_ep_info is not None:
                self.ep_info_buffer.extend([maybe_ep_info])
            if maybe_is_success is not None and dones[idx]:
                self.ep_success_buffer.append(maybe_is_success)
            if maybe_removal_success is not None and maybe_removal_done:
                self.removal_success_buffer.append(maybe_removal_success)
            if maybe_global_success is not None and maybe_global_done:
                self.global_success_buffer.append(maybe_global_success)

    def _upper_env_update_info_buffer(self, infos, dones):
        assert self.upper_info_buffer is not None
        assert dones is not None

        for idx, info in enumerate(infos):
            upper_info = info.get('upper_info')
            assert upper_info is not None
            self.upper_info_buffer.extend([upper_info])


import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

lower = 0
upper = 1
indicate_list = [lower, upper]


def make_env(env_name, agent_path=None, device=None):
    def _thunk():
        if agent_path is not None:
            env = gym.make(env_name, agent_path=agent_path, device=device)
        else:
            env = gym.make(env_name)
        env = Monitor(env, None, allow_early_resets=True)

        return env

    return _thunk


def env_wrapper(env_name, num_envs, agent_path=None, device=None):
    envs = [
        make_env(env_name, agent_path=agent_path, device=device)
        for _ in range(num_envs)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecNormalize(envs, norm_reward=True, norm_obs=False, training=False)

    return envs


class ExecutePPO:
    def __init__(
            self,

            lower_policy: Union[str, Union[Type[ActorCriticPolicy], Type[HybridPolicy]]],
            upper_policy: Union[str, Union[Type[ActorCriticPolicy], Type[HybridPolicy]]],

            lower_env: Union[GymEnv, str],
            upper_env: Union[GymEnv, str],

            lower_env_id: str = 'ExecuteDense-v0',
            lower_env_num: int = 40,
            lower_n_steps: int = 2048,
            lower_batch_size: int = 128,

            upper_env_id: str = 'Plan-v0',
            upper_env_num: int = 3,
            upper_n_steps: int = 64,
            upper_batch_size: int = 16,

            learning_rate: Union[float, Schedule] = 3e-4,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: Union[float, Schedule] = 0.2,
            clip_range_vf: Union[None, float, Schedule] = None,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            target_kl: Optional[float] = None,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        self.lower_agent = HybridPPO(lower_policy, lower_env, learning_rate, lower_n_steps,
                                     lower_batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, ent_coef,
                                     vf_coef, max_grad_norm, use_sde, sde_sample_freq, target_kl, tensorboard_log,
                                     create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model,
                                     is_two_stage_env=False,
                                     upper_counting_mode=False,
                                     )
        self.upper_agent = HybridPPO(upper_policy, upper_env, learning_rate, upper_n_steps,
                                     upper_batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, ent_coef,
                                     vf_coef, max_grad_norm, use_sde, sde_sample_freq, target_kl, tensorboard_log,
                                     create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model,
                                     is_two_stage_env=False,
                                     upper_counting_mode=False,
                                     )

        self.lower_policy = lower_policy
        self.lower_env_id = lower_env_id
        self.lower_env_num = lower_env_num
        self.lower_n_steps = lower_n_steps
        self.lower_batch_size = lower_batch_size

        self.upper_policy = upper_policy
        self.upper_env_id = upper_env_id
        self.upper_env_num = upper_env_num
        self.upper_n_steps = upper_n_steps
        self.upper_batch_size = upper_batch_size

        self.device = device
        self.seed = seed
        self.lr = learning_rate
        self.target_kl = target_kl
        self.tensorboard_log = tensorboard_log

    def load_lower(self, agent_path: str = None, logger=None):
        assert logger is not None, 'Logger can not be None!!!'
        wrapped_lower_env = env_wrapper(self.lower_env_id, self.lower_env_num, agent_path=agent_path,
                                        device=self.device)
        self.lower_agent = HybridPPO(self.lower_policy,
                                     wrapped_lower_env,
                                     self.lr,
                                     self.lower_n_steps,
                                     batch_size=self.lower_batch_size,
                                     verbose=1,
                                     tensorboard_log=self.tensorboard_log,
                                     device=self.device,
                                     is_two_stage_env=False,
                                     target_kl=self.target_kl,
                                     seed=self.seed,
                                     )
        self.lower_agent.set_logger(logger)

    def load_upper(self, agent_path: str = None, logger=None):
        assert logger is not None, 'Logger can not be None!!!'
        wrapped_upper_env = env_wrapper(self.upper_env_id, self.upper_env_num, agent_path=agent_path,
                                        device=self.device)
        self.upper_agent = HybridPPO(self.upper_policy,
                                     wrapped_upper_env,
                                     self.lr,
                                     self.upper_n_steps,
                                     batch_size=self.upper_batch_size,
                                     verbose=1,
                                     tensorboard_log=self.tensorboard_log,
                                     device=self.device,
                                     is_two_stage_env=False,
                                     upper_counting_mode=True,
                                     target_kl=self.target_kl,
                                     seed=self.seed,
                                     )
        self.upper_agent.set_logger(logger)

    def learn(
            self,
            total_iteration_count: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "PPO",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            lower_save_interval: Optional[int] = None,
            lower_save_path: Optional[str] = None,
            upper_save_interval: Optional[int] = None,
            upper_save_path: Optional[str] = None,
            save_count: int = 0,
            train_lower_iteration: int = 100,
            train_upper_iteration: int = 3000,
            train_lower_model_path: str = None,
            train_upper_model_path: str = None,
    ):
        lower_save_count = save_count
        lower_iteration = 0
        lower_total_timesteps = 0
        lower_time_elapsed = 0

        upper_save_count = save_count
        upper_iteration = 0
        upper_total_timesteps = 0
        upper_time_elapsed = 0

        start_time = time.time()

        # judge whether model_path provided or not
        is_lower_model_provided = train_lower_model_path is not None
        is_upper_model_provided = train_upper_model_path is not None

        if is_lower_model_provided:
            self.load_agent(env=self.lower_agent.env, agent_name='lower', model_path=train_lower_model_path)
        if is_upper_model_provided:
            self.load_agent(env=self.upper_agent.env, agent_name='upper', model_path=train_upper_model_path)

        lower_single_steps = self.lower_agent.rollout_buffer.n_envs * self.lower_agent.rollout_buffer.buffer_size
        lower_total_steps = lower_single_steps * total_iteration_count
        lower_total_steps, lower_callback = self.lower_agent.setup_learn(total_timesteps=lower_total_steps,
                                                                         callback=callback,
                                                                         eval_env=eval_env,
                                                                         eval_freq=eval_freq,
                                                                         n_eval_episodes=n_eval_episodes,
                                                                         eval_log_path=eval_log_path,
                                                                         reset_num_timesteps=reset_num_timesteps,
                                                                         tb_log_name=f'Lower')

        upper_single_steps = self.upper_agent.rollout_buffer.n_envs * self.upper_agent.rollout_buffer.buffer_size
        upper_total_steps = upper_single_steps * total_iteration_count
        upper_total_steps, upper_callback = self.upper_agent.setup_learn(total_timesteps=upper_total_steps,
                                                                         callback=callback,
                                                                         eval_env=eval_env, eval_freq=eval_freq,
                                                                         n_eval_episodes=n_eval_episodes,
                                                                         eval_log_path=eval_log_path,
                                                                         reset_num_timesteps=reset_num_timesteps,
                                                                         tb_log_name=f'Upper')

        lower_callback.on_training_start(locals(), globals())
        upper_callback.on_training_start(locals(), globals())

        latest_lower_model_path = f'{lower_save_path}_{lower_save_count}'
        latest_upper_model_path = f'{upper_save_path}_{upper_save_count}'
        self.lower_agent.save(latest_lower_model_path)
        self.upper_agent.save(latest_upper_model_path)

        for iteration in range(total_iteration_count):
            print(f'Round {iteration + 1} training starts!')

            if not is_lower_model_provided:
                self.load_lower(latest_upper_model_path, self.upper_agent.logger)
                lower_single_steps = self.lower_agent.rollout_buffer.n_envs * self.lower_agent.rollout_buffer.buffer_size
                self.lower_agent.set_logger(self.lower_agent.logger)
                self.lower_agent.learn_one_step(train_lower_iteration * lower_single_steps,
                                                lower_callback, log_interval, eval_env, eval_freq, n_eval_episodes,
                                                f'Lower', eval_log_path, reset_num_timesteps, lower_save_interval,
                                                lower_save_path,
                                                accumulated_save_count=lower_save_count,
                                                accumulated_time_elapsed=lower_time_elapsed,
                                                accumulated_iteration=lower_iteration,
                                                accumulated_total_timesteps=lower_total_timesteps,
                                                prefix='Lower')
                lower_save_count += train_lower_iteration // lower_save_interval
                lower_iteration += train_lower_iteration
                lower_time_elapsed += time.time() - self.lower_agent.start_time
                lower_total_steps += self.lower_agent.num_timesteps
                latest_lower_model_path = f'{lower_save_path}_{lower_save_count}'
            else:
                latest_lower_model_path = train_lower_model_path

            if not is_upper_model_provided:
                self.load_upper(latest_lower_model_path, self.upper_agent.logger)
                upper_single_steps = self.upper_agent.rollout_buffer.n_envs * self.upper_agent.rollout_buffer.buffer_size
                self.upper_agent.set_logger(self.upper_agent.logger)
                self.upper_agent.learn_one_step(train_upper_iteration * upper_single_steps,
                                                callback, log_interval, eval_env, eval_freq, n_eval_episodes,
                                                f'Upper', eval_log_path, reset_num_timesteps, upper_save_interval,
                                                upper_save_path,
                                                accumulated_save_count=upper_save_count,
                                                accumulated_time_elapsed=upper_time_elapsed,
                                                accumulated_iteration=upper_iteration,
                                                accumulated_total_timesteps=upper_total_timesteps,
                                                prefix='Upper')
                upper_save_count += train_upper_iteration // upper_save_interval
                upper_iteration += train_upper_iteration
                upper_time_elapsed += time.time() - self.upper_agent.start_time
                upper_total_timesteps += self.upper_agent.num_timesteps
                latest_upper_model_path = f'{upper_save_path}_{upper_save_count}'
            else:
                latest_upper_model_path = train_upper_model_path

            print(f'Round {iteration + 1} training ends!')
            print('-' * 64 + f' Total Time Elapsed: {time.time() - start_time} ' + '-' * 64)

        lower_callback.on_training_end()
        upper_callback.on_training_end()

    def load_agent(self, env: gym.Env = None, agent_name: str = None, model_path: str = None):
        assert env is not None, 'Env can not be None!'
        assert agent_name is not None, 'Agent name can not be None!'
        assert model_path is not None, 'Model path can not be None!'
        agent_name_list = ['lower', 'upper']

        if agent_name == 'lower':
            self.lower_agent.load(model_path, env=env, device=self.device, seed=self.seed,
                                  is_two_stage_env=True, upper_counting_mode=False)
            self.lower_agent.reset_rollout_buffer(env)
        elif agent_name == 'upper':
            self.upper_agent.load(model_path, env=env, device=self.device, seed=self.seed,
                                  is_two_stage_env=False, upper_counting_mode=True)
            self.upper_agent.reset_rollout_buffer(env)
        else:
            assert False, f'Agent name is invalid!\nAgent name must in {agent_name_list}'

    def load_all_agent(self, model_dir: str = None):
        agent_name_list = ['lower', 'upper']
        agent_list = [self.lower_agent, self.upper_agent]
        postfix = '.zip'
        for i in range(len(agent_name_list)):
            for model_path in os.listdir(model_dir):
                if agent_name_list[i] in model_path:
                    self.load_agent(agent_list[i].env, agent_name_list[i],
                                    os.path.join(model_dir, model_path.replace(postfix, '')))
                    break
            assert isinstance(agent_list[i], HybridPPO), f'{agent_name_list[i]} agent load failed!'
