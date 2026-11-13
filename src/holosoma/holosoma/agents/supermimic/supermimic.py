"""Online model-based actor-critic for G1 velocity tracking."""

from __future__ import annotations

import copy
import itertools
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.agents.supermimic.config import SuperMimicConfig
from holosoma.agents.supermimic.history import WindowReservoir
from holosoma.agents.supermimic.kinematics import LegGeometry
from holosoma.agents.supermimic.locomotion import LocomotionModelIO, SuperMimicLocomotion
from holosoma.agents.supermimic.world_model import Dynamics, SequenceReplay, imagination_step, lambda_returns, mlp
from holosoma.agents.supermimic.reference_components import RunningMoments
from holosoma.agents.supermimic.reference_dynamics import ReferenceDynamics
from holosoma.agents.supermimic.reference_optim import build_reference_optimizer
from holosoma.agents.supermimic.reference_actor import VelocityActor
from torch.utils.checkpoint import checkpoint as recompute
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
)
from holosoma.utils.safe_torch_import import F, TensorboardSummaryWriter, nn, optim, torch


class Actor(nn.Module):
    def __init__(
        self, obs_dim: int, hidden_dim: int, action_bound: torch.Tensor, obs_action_scale: torch.Tensor | None = None,
        reference: bool = False,
        raw_actions: bool = False,
    ):
        super().__init__()
        self.network = mlp(obs_dim, hidden_dim, action_bound.numel(), nn.GELU if reference else nn.SiLU)
        self.normalizer = RunningMoments(obs_dim) if reference else nn.Identity()
        self.raw_actions = raw_actions
        if not reference:
            nn.init.uniform_(self.network[-1].weight, -1e-3, 1e-3)
            nn.init.zeros_(self.network[-1].bias)
        self.register_buffer("action_bound", action_bound)
        self.register_buffer("obs_action_scale", obs_action_scale, persistent=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_action_scale is not None:
            obs = obs * self.obs_action_scale
        output = self.network(self.normalizer(obs))
        return output if self.raw_actions else output.tanh() * self.action_bound


class SuperMimicAgent(BaseAlgo):
    env: SuperMimicLocomotion
    config: SuperMimicConfig
    MODEL_FIELDS = ("state", "next_state", "applied_action", "context", "done", "terminated")
    SEED_FIELDS = ("state", "context", "command", "phase", "phase_dt", "previous_action", "delayed")

    def __init__(
        self,
        env: BaseTask,
        config: SuperMimicConfig,
        device: str,
        log_dir: str,
        multi_gpu_cfg: dict[str, int] | None = None,
    ):
        if not isinstance(env, SuperMimicLocomotion) or env.num_dof != 29:
            raise ValueError("SuperMimic requires the G1 29-DoF locomotion task.")
        if env.simulator.simulator_config.name != "isaacgym":
            raise ValueError("This SuperMimic experiment requires Isaac Gym Preview 4.")
        super().__init__(env, config, device, multi_gpu_cfg)
        if self.is_multi_gpu:
            raise ValueError("SuperMimic currently supports one training device.")
        if not 1 <= config.model_horizon <= config.replay_steps:
            raise ValueError("SuperMimic requires 1 <= model_horizon <= replay_steps.")
        if not 0 <= config.model_history_fraction < 1:
            raise ValueError("WM history fraction must be in [0, 1), retaining recent supervision.")
        if config.model_history_fraction and not (
            config.model_horizon <= config.history_window <= config.replay_steps
            and min(config.history_capacity, config.history_holdout_capacity, config.history_candidates) > 0
        ):
            raise ValueError("History needs positive capacities and model_horizon <= history_window <= replay_steps.")
        if (
            min(
                config.num_steps_per_env,
                config.batch_size,
                config.imagination_horizon,
                config.critic_epochs,
                config.model_updates,
                config.logging_interval,
                config.save_interval,
            )
            < 1
        ):
            raise ValueError("SuperMimic rollout sizes, critic epochs and logging intervals must be positive.")
        if not 0 <= config.real_critic_weight <= 1 or (config.real_critic_weight and config.replay_steps <= config.model_horizon):
            raise ValueError("The real critic needs a weight in [0, 1] and room for one more replay frame.")
        if config.holdout_stride and (config.holdout_stride < 2 or env.num_envs < 2):
            raise ValueError("The holdout needs at least one training world and one validation world.")
        if not 0 <= config.exploration_correlation < 1:
            raise ValueError("Exploration correlation must be in [0, 1).")
        self.log_dir = Path(log_dir)
        self.iteration = 0
        self.collected_steps = 0
        if config.implementation not in ("legacy_v4", "public_gencoord_v1", "public_gencoord_v2", "public_gencoord_v3"):
            raise ValueError(f"Unknown SuperMimic implementation: {config.implementation}")
        self.reference = config.implementation != "legacy_v4"
        self.raw_actions = config.implementation in ("public_gencoord_v2","public_gencoord_v3")
        if self.reference and config.model_history_fraction:
            raise ValueError("Public reference uses FIFO, not a fixed historical mixture")
        if self.reference:
            self.MODEL_FIELDS = (*self.MODEL_FIELDS, "root_pose", "next_root_pose")

    def setup(self) -> None:  # type: ignore[override]
        cfg = self.config
        self.io = LocomotionModelIO(self.env)
        self.io.reward()  # Validate the selected reward before collecting data.
        state_dim = self.io.state_scale.numel()
        action_dim = self.io.action_bound.numel()
        context_dim = self.io.context().shape[1]
        obs_action_scale = self.io.obs_action_scale if cfg.normalize_action_history and not self.raw_actions else None
        if cfg.implementation=="public_gencoord_v3":
            goal_indices=[index for name in ("command_lin_vel","command_ang_vel")
                for index in range(self.io.term_slices[name].start,self.io.term_slices[name].stop)]
            self.actor=VelocityActor(self.io.obs_scale.numel(),goal_indices,self.io.obs_scale,self.io.action_bound,
                cfg.actor_hidden_dims,cfg.actor_latent_dim).to(self.device)
        else:
            self.actor = Actor(self.io.obs_scale.numel(), cfg.actor_hidden_dim, self.io.action_bound,
                               obs_action_scale, self.reference, self.raw_actions).to(self.device)
        _, urdf = get_urdf_text_from_robot_config(self.env.robot_config)
        geometry = LegGeometry(urdf, self.env.robot_config.dof_names,
            self.env.default_dof_pos_base.squeeze(0), ["left_ankle_roll_link", "right_ankle_roll_link"])
        if self.reference:
            self.world_model = ReferenceDynamics(self.io.state_scale,
                self.io.action_bound*self.io.control.action_scales,
                self.env.default_dof_pos_base.squeeze(0),context_dim,cfg.model_hidden_dim,
                geometry,self.env.dt,round(self.env.dt/self.env.sim_dt)).to(self.device)
        else:
            self.world_model = Dynamics(
                state_dim, action_dim, context_dim, cfg.model_hidden_dim,
                joint_position_start=sum(self.io.state_sizes[:3]),
                joint_dt_scaled=self.env.dt / 0.05 if cfg.kinematic_joint_prior else 0.0,
                geometry=geometry,
            ).to(self.device)
        activation = nn.GELU if self.reference else nn.SiLU
        # Command (3), sine and cosine of two leg phases (4), phase increment and delay.
        self.critic = nn.Sequential(
            nn.Linear(state_dim + context_dim + action_dim + 9, cfg.critic_hidden_dim),
            nn.LayerNorm(cfg.critic_hidden_dim), activation(),
            nn.Linear(cfg.critic_hidden_dim, cfg.critic_second_hidden_dim),
            nn.LayerNorm(cfg.critic_second_hidden_dim), activation(),
            nn.Linear(cfg.critic_second_hidden_dim, 1),
        ).to(self.device)
        if not self.reference:
            nn.init.zeros_(self.critic[-1].weight)
            nn.init.zeros_(self.critic[-1].bias)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.learning_rate, betas=(0.9, cfg.actor_beta2))
        self.model_optimizer = optim.Adam(self.world_model.parameters(), lr=cfg.model_learning_rate or cfg.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        if self.reference:
            self.actor_optimizer = build_reference_optimizer(self.actor,cfg.learning_rate,(.9,cfg.actor_beta2))
            self.model_optimizer = build_reference_optimizer(self.world_model,cfg.model_learning_rate or cfg.learning_rate)
            self.critic_optimizer = build_reference_optimizer(self.critic,cfg.learning_rate)
        self.replay = SequenceReplay(cfg.replay_steps, cfg.replay_storage_device)
        self.exploration_state = self.io.action_bound.new_zeros(self.env.num_envs, action_dim)
        self.exploration_std = (
            self.io.action_bound.new_full((action_dim,), cfg.exploration_noise)
            if cfg.exploration_joint_std is None
            else cfg.exploration_joint_std / self.io.control.action_scales
        )
        self.train_worlds = self.holdout_worlds = None
        if cfg.holdout_stride:
            worlds = torch.arange(self.env.num_envs, device=self.device)
            holdout = worlds.remainder(cfg.holdout_stride) == 0
            self.train_worlds, self.holdout_worlds = worlds[~holdout], worlds[holdout]
        self.history = self.history_holdout = None
        if cfg.model_history_fraction:
            self.history = WindowReservoir(cfg.history_capacity, cfg.history_window, cfg.history_seed)
            if self.holdout_worlds is not None:
                self.history_holdout = WindowReservoir(cfg.history_holdout_capacity, cfg.history_window, cfg.history_seed + 2)
        self.writer = TensorboardSummaryWriter(log_dir=str(self.log_dir), flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            self.env.num_envs,
            cfg.num_steps_per_env,
            cfg.num_learning_iterations,
            device=self.device,
            title="SuperMimic",
        )

    @torch.no_grad()
    def _collect(self, obs: dict[str, torch.Tensor], steps: int | None = None) -> dict[str, torch.Tensor]:
        env, io = self.env, self.io
        gait = io.gait
        correlation = self.config.exploration_correlation
        innovation_scale = math.sqrt(1.0 - correlation**2)
        assert gait.phase is not None and gait.phase_dt is not None
        steps = self.config.num_steps_per_env if steps is None else steps
        for _ in range(steps):
            bound = io.action_bound
            if self.reference:
                actor_obs = obs["actor_obs"].clone()
                if self.actor.obs_action_scale is not None:
                    actor_obs = actor_obs*self.actor.obs_action_scale
                if not self.raw_actions:  # Preserve the superseded v1 collection convention.
                    self.actor.normalizer.update(actor_obs if self.train_worlds is None else actor_obs[self.train_worlds])
            action = self.actor(obs["actor_obs"])
            self.exploration_state.mul_(correlation).add_(torch.randn_like(action), alpha=innovation_scale)
            action = action + self.exploration_std * self.exploration_state
            if not self.raw_actions:
                action = action.clamp(-bound, bound)
            transition = {
                "state": io.state(),
                "context": io.context(),
                "command": env.command_manager.commands.clone(),
                "phase": gait.phase.clone(),
                "phase_dt": gait.phase_dt.clone(),
                "previous_action": env.action_manager.action.clone(),
                "delayed": io.delay(),
            }
            if self.reference:
                transition["root_pose"] = io.root_pose()
            obs, reward, done, infos = env.step({"actions": action})
            assert env.model_successor is not None
            state, applied_action = env.model_successor
            episode_done = done.bool()
            if correlation:
                self.exploration_state.masked_fill_(episode_done[:, None], 0.0)
            model_done = episode_done
            if env.model_push is not None:
                model_done = episode_done.clone()
                model_done[env.model_push[0]] = True
            transition.update(
                next_state=state,
                applied_action=applied_action / bound,
                done=model_done,
                terminated=episode_done & ~infos["time_outs"].bool(),
            )
            transition.update(reward=reward, episode_done=episode_done)
            if self.reference:
                transition["next_root_pose"] = env.model_root_successor
                worlds = slice(None) if self.train_worlds is None else self.train_worlds
                self.world_model.update_normalizers(transition["state"][worlds],transition["applied_action"][worlds])
                if self.raw_actions:
                    # Public runner flushes fresh-data moments after stepping;
                    # computing the action never changes its own input scaling.
                    self.actor.normalizer.update(actor_obs[worlds])
            self.replay.append(transition)
            self.collected_steps += 1
            if self.history is not None and self.replay.steps % self.config.history_window == 0:
                worlds = self.train_worlds
                if worlds is None:
                    worlds = torch.arange(env.num_envs)
                self.history.capture(self.replay, self.MODEL_FIELDS, worlds,
                                     self.config.history_candidates, self.collected_steps)
                if self.history_holdout is not None:
                    self.history_holdout.capture(self.replay, self.MODEL_FIELDS, self.holdout_worlds,
                        max(1, self.config.history_candidates // 4), self.collected_steps)
            self.logging_helper.update_episode_stats(reward, done, infos)
        return obs

    def _fit_model(self) -> dict[str, torch.Tensor]:
        cfg = self.config
        size = cfg.model_batch_size or cfg.batch_size
        historical = int(size * cfg.model_history_fraction) if self.history is not None and len(self.history) else 0
        batch = self.replay.sample(size - historical, cfg.model_horizon, self.MODEL_FIELDS, self.train_worlds,
                                   **({"complete": True} if self.reference else {}))
        if historical:
            old = self.history.sample(historical, cfg.model_horizon, self.MODEL_FIELDS, self.device)
            batch = {name: torch.cat((batch[name], old[name])) for name in self.MODEL_FIELDS}
        state_loss, fall_loss = self.world_model.rollout_loss(batch)
        # Eq. (1) uses a squared vector norm, not a per-coordinate average.
        # Keep the reported MSE interpretable without shrinking its training weight.
        self._optimize(state_loss * (1 if self.reference else self.io.state_scale.numel()) + fall_loss,
                       self.model_optimizer, self.world_model)
        return {"model_state_mse": state_loss.detach(), "model_fall_bce": fall_loss.detach(),
                "model_history_fraction": state_loss.new_tensor(historical / size)}

    @torch.no_grad()
    def _validate_model(self) -> dict[str, torch.Tensor]:
        cfg = self.config
        batch = self.replay.sample(cfg.model_batch_size or cfg.batch_size, cfg.model_horizon, self.MODEL_FIELDS, self.holdout_worlds)
        loss, fall_loss = self.world_model.rollout_loss(batch)
        valid = torch.cat((torch.ones_like(batch["done"][:, :1]), (~batch["done"][:, :-1]).cumprod(1).bool()), dim=1)
        squared_error = (batch["state"][:, :1] - batch["next_state"]).square().mean(dim=-1)
        if self.reference:
            squared_error = self.world_model.physical_error(batch["state"][:, :1],batch["root_pose"][:, :1],
                batch["next_state"],batch["next_root_pose"])
        persistence = (squared_error * valid).sum() / valid.sum()
        successor, _ = self.world_model(batch["state"][:, 0], batch["applied_action"][:, 0], batch["context"][:, 0])
        velocity_error = (successor[:, :3] - batch["next_state"][:, 0, :3]) / self.io.state_scale[:3]
        metrics = {
            "holdout_state_mse": loss,
            "holdout_fall_bce": fall_loss,
            "holdout_persistence_mse": persistence,
            "holdout_velocity_rmse_mps": velocity_error.square().mean().sqrt(),
        }
        if self.history is not None:
            for key, value in self.history.coverage(self.collected_steps, cfg.replay_steps).items():
                metrics["history_" + key] = loss.new_tensor(value)
        if self.history_holdout is not None and len(self.history_holdout):
            old = self.history_holdout.sample(cfg.model_batch_size or cfg.batch_size, cfg.model_horizon,
                                              self.MODEL_FIELDS, self.device)
            metrics["history_holdout_state_mse"], metrics["history_holdout_fall_bce"] = self.world_model.rollout_loss(old)
        return metrics

    def _value_input(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        command: torch.Tensor,
        phase: torch.Tensor,
        phase_dt: torch.Tensor,
        previous_action: torch.Tensor,
        delayed: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                state,
                context,
                command,
                phase.sin(),
                phase.cos(),
                phase_dt,
                previous_action / self.io.action_bound,
                delayed,
            ),
            dim=-1,
        )

    def _imagine_policy(self, seeds: dict[str, torch.Tensor], with_noise: bool = True) -> dict[str, Any]:
        """Shared training/diagnostic objective; callers freeze model/critic weights.

        Keep their input Jacobians, including the terminal value and survival.
        Seed tensors have the same [batch, 1, features] layout as replay samples.
        """
        cfg, io = self.config, self.io
        state, context, command, phase, phase_dt, previous_action, delayed = (
            seeds[name][:, 0] for name in self.SEED_FIELDS
        )
        reward_fn = io.reward()
        stand = (command[:, :2].norm(dim=-1, keepdim=True) < 0.01) & (command[:, 2:3].abs() < 0.01)
        value_inputs, rewards, discounts = [], [], []
        weight = torch.ones_like(command[:, 0])
        objective = torch.zeros_like(weight)

        physical = io.unpack(state)
        for _ in range(cfg.imagination_horizon):
            value_input = self._value_input(state, context, command, phase, phase_dt, previous_action, delayed)
            value_inputs.append(value_input)
            observation = io.actor_observation(physical, command, phase, previous_action, with_noise=with_noise)
            if cfg.implementation=="public_gencoord_v3" and cfg.checkpoint_imagination:
                action = recompute(self.actor,observation,use_reentrant=False,preserve_rng_state=False)
            else:
                action = self.actor(observation)
            applied_action = io.applied_action(action, previous_action, delayed)
            state, fall_logit = imagination_step(self.world_model, state, applied_action, context,
                                                cfg.checkpoint_imagination)
            physical = io.unpack(state)
            # Hold the sampled velocity command and gait frequency over this short rollout.
            moving_phase = torch.remainder(phase + phase_dt + torch.pi, 2 * torch.pi) - torch.pi
            next_phase = torch.where(stand, io.gait.stand_phase_value, moving_phase)
            reward_phase = next_phase if io.is_isaacgym else phase
            reward = reward_fn(physical, command, reward_phase, action, previous_action)
            discount = cfg.gamma * torch.sigmoid(-fall_logit)
            if cfg.detach_fall_discount:
                discount = discount.detach()
            rewards.append(reward.detach())
            discounts.append(discount.detach())
            objective = objective + weight * reward
            weight = weight * discount
            previous_action = action
            phase = next_phase

        last_input = self._value_input(state, context, command, phase, phase_dt, previous_action, delayed)
        bootstrap = weight * self._bounded_value(last_input)
        return dict(objective=objective + bootstrap, reward_objective=objective, bootstrap=bootstrap,
                    value_inputs=value_inputs, rewards=rewards, discounts=discounts, last_input=last_input)

    def _value_ceiling(self):
        # All positive G1 reward terms supported by LocomotionReward are <= 1.
        reward = self.io.reward()
        return sum(max(0.,term.weight) for term in reward.terms.values())*self.env.dt/(1-self.config.gamma)

    def _bounded_value(self, inputs):
        values = self.critic(inputs).squeeze(-1)
        return values.clamp_max(self._value_ceiling()) if self.reference else values

    def _fit_policy(self) -> dict[str, torch.Tensor]:
        cfg = self.config
        seeds = self.replay.sample(cfg.batch_size, 1, self.SEED_FIELDS, self.train_worlds)
        # Keep the input Jacobians of both networks for BPTT.
        self.world_model.requires_grad_(False)
        self.critic.requires_grad_(False)
        try:
            rollout = self._imagine_policy(seeds)
            actor_loss = -rollout["objective"].mean()
            actor_grad = self._optimize(actor_loss, self.actor_optimizer, self.actor)
        finally:
            self.world_model.requires_grad_(True)
            self.critic.requires_grad_(True)

        with torch.no_grad():
            value_inputs, last_input = rollout["value_inputs"], rollout["last_input"]
            rewards, discounts = rollout["rewards"], rollout["discounts"]
            value_inputs.append(last_input)
            inputs = torch.stack(value_inputs)
            # The release clamps completed GAE targets, not every intermediate
            # value in the recursion; only actor bootstrapping clamps V itself.
            next_values = self.critic(inputs[1:]).squeeze(-1)
            targets = lambda_returns(torch.stack(rewards), torch.stack(discounts), next_values, cfg.gae_lambda)
            if self.reference:
                targets = targets.clamp_max(self._value_ceiling())
            if cfg.real_critic_weight:
                real_inputs, real_targets = self._real_critic_batch()
        critic_losses = []
        flat_inputs = inputs[:-1].flatten(0, 1)
        flat_targets = targets.flatten()
        permutation = None
        for _ in range(cfg.critic_epochs):
            if permutation is None or not self.reference:
                permutation = torch.randperm(flat_targets.numel(), device=self.device)
            for indices in permutation.tensor_split(cfg.critic_minibatches):
                critic_loss = F.mse_loss(self.critic(flat_inputs[indices]).squeeze(-1), flat_targets[indices])
                if cfg.real_critic_weight:
                    real_loss = F.mse_loss(self.critic(real_inputs).squeeze(-1), real_targets)
                    critic_loss = torch.lerp(critic_loss, real_loss, cfg.real_critic_weight)
                self._optimize(critic_loss, self.critic_optimizer, self.critic)
                critic_losses.append(critic_loss.detach())
        return {
            "actor": actor_loss.detach(),
            "actor_grad_norm": actor_grad.detach(),
            "critic": torch.stack(critic_losses).mean(),
            "imagined_step_reward": torch.stack(rewards).mean(),
        }

    def _real_critic_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        batch = self.replay.sample(
            cfg.batch_size, cfg.model_horizon + 1, (*self.SEED_FIELDS, "reward", "episode_done"), self.train_worlds
        )
        inputs = self._value_input(*(batch[name].transpose(0, 1) for name in self.SEED_FIELDS))
        rewards = batch["reward"][:, :-1].transpose(0, 1)
        discounts = cfg.gamma * (~batch["episode_done"][:, :-1].transpose(0, 1))
        targets = lambda_returns(rewards, discounts, self.critic(inputs[1:]).squeeze(-1), cfg.gae_lambda)
        return inputs[:-1], targets

    def _optimize(self, loss: torch.Tensor, optimizer: optim.Optimizer, model: nn.Module) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
        if optimizer is self.actor_optimizer:
            nn.utils.clip_grad_value_(model.parameters(), self.config.actor_grad_clip_value)
        optimizer.step()
        return norm

    def learn(self) -> None:  # type: ignore[override]
        cfg = self.config
        if self.iteration >= cfg.num_learning_iterations:
            logger.info("SuperMimic checkpoint already reached the requested iteration.")
            self.writer.close()
            return

        torch.set_float32_matmul_precision("high")
        obs = self.env.reset_all()
        last_logged = self.iteration
        last_logged_steps = self.collected_steps
        self.logging_helper.tot_timesteps = self.collected_steps * self.env.num_envs
        self.env.model_io = self.io
        model_ready = self.holdout_worlds is None or not cfg.model_readiness_gate
        logger.info(
            f"SuperMimic collecting from {self.env.num_envs} worlds at {self.env.dt * 1000:g} ms per control step, "
            f"with {cfg.model_horizon} replay steps and {cfg.imagination_horizon} imagined steps."
        )
        try:
            for iteration in range(self.iteration + 1, cfg.num_learning_iterations + 1):
                final = iteration == cfg.num_learning_iterations
                with self.logging_helper.record_collection_time():
                    ratio = cfg.update_to_data_ratio
                    steps = math.floor(iteration / ratio) - math.floor((iteration - 1) / ratio)
                    obs = self._collect(obs, steps=steps)
                progress = min(iteration / cfg.lr_decay_iterations, 1.0)
                factor = cfg.lr_final_fraction + (1 - cfg.lr_final_fraction) * (1 + math.cos(math.pi * progress)) / 2
                if self.reference:
                    factor = 1.0  # Public release uses constant LR; paper used cosine Adam.
                for optimizer, rate in ((self.actor_optimizer, cfg.learning_rate),
                                        (self.critic_optimizer, cfg.learning_rate),
                                        (self.model_optimizer, cfg.model_learning_rate or cfg.learning_rate)):
                    for group in optimizer.param_groups:
                        group["lr"] = rate * factor
                metrics = {}
                with self.logging_helper.record_learn_time():
                    if len(self.replay) >= cfg.model_horizon:
                        for _ in range(cfg.model_updates):
                            metrics.update(self._fit_model())
                        if self.holdout_worlds is not None and (iteration % cfg.logging_interval == 0 or final):
                            metrics.update(self._validate_model())
                            if cfg.model_readiness_gate:
                                was_ready = model_ready
                                model_ready = bool(metrics["holdout_state_mse"] < 0.9 * metrics["holdout_persistence_mse"])
                                if was_ready != model_ready:
                                    logger.info(f"World model held-out persistence gate at iteration {iteration}: {model_ready}.")
                        enough_replay = len(self.replay) > cfg.model_horizon or not cfg.real_critic_weight
                        if iteration > cfg.policy_warmup_iterations and model_ready and enough_replay:
                            metrics.update(self._fit_policy())
                self.iteration = iteration
                if final or iteration % cfg.logging_interval == 0:
                    self.logging_helper.num_steps_per_env = self.collected_steps - last_logged_steps
                    last_logged_steps = self.collected_steps
                    self.logging_helper.post_epoch_logging(
                        iteration, {name: value.item() for name, value in metrics.items()}, {}
                    )
                    last_logged = iteration
                if final or iteration % cfg.save_interval == 0:
                    self.save(name=f"model_{iteration}.pt")
        finally:
            self.env.model_io = None
            self.env.model_successor = None
            self.writer.close()

    def save(self, path: str | None = None, name: str = "last.pt") -> None:
        state = {
            "format_version": 5 if self.reference else 4,
            "implementation": self.config.implementation,
            "official_reference_commit": "5efbf114e11f725c31d83e217be987c13c0e4ce2" if self.reference else None,
            "physics_context_encoding": self.io.context_encoding,
            "collected_steps": self.collected_steps,
            "actor": self.actor.state_dict(),
            "world_model": self.world_model.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "env_state": self._collect_env_state(),
            **self._checkpoint_metadata(self.iteration),
        }
        if self.history is not None:
            state["model_history"] = dict(num_envs=self.env.num_envs, holdout_stride=self.config.holdout_stride,
                train=self.history.state_dict(),
                heldout=self.history_holdout.state_dict() if self.history_holdout is not None else None)
        self.logging_helper.save_checkpoint_artifact(state, path or str(self.log_dir / name))

    def load(self, path: str, load_history: bool = True) -> None:  # type: ignore[override]
        # History is CPU-resident; loading it onto the GPU first can exhaust VRAM.
        state = torch.load(path, map_location="cpu")
        if state.get("format_version") not in (3, 4, 5):
            raise ValueError("Unsupported SuperMimic checkpoint format")
        if state.get("implementation", "legacy_v4") != self.config.implementation:
            raise ValueError("Checkpoint implementation differs; restore its saved config before setup")
        if state["format_version"] == 3:
            encoding = "isaacgym_batch_centered_v0"
        else:
            encoding = state["physics_context_encoding"]
        self.io.set_context_encoding(encoding)
        if encoding == "isaacgym_batch_centered_v0":
            logger.warning("Legacy checkpoint retains batch-centered physics context; its training reference was not saved. "
                           "Fresh training uses a fixed physical reference.")
        self.collected_steps = state["collected_steps"]
        self.actor.load_state_dict(state["actor"])
        self.world_model.load_state_dict(state["world_model"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.model_optimizer.load_state_dict(state["model_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.iteration = state["iteration"]
        self._restore_env_state(state.get("env_state"))
        if load_history and self.history is not None:
            saved = state.get("model_history")
            if saved is None:
                logger.warning("Checkpoint has no WM history: reservoir starts empty; this is not an exact training continuation.")
            else:
                if (saved["num_envs"], saved["holdout_stride"]) != (self.env.num_envs, self.config.holdout_stride):
                    raise ValueError("Cannot resume history with different world/holdout assignments; use load_history=False for evaluation.")
                self.history.load_state_dict(saved["train"])
                if self.history_holdout is not None:
                    self.history_holdout.load_state_dict(saved["heldout"])
            # Recent replay is deliberately empty after reset, preventing windows
            # across two unrelated simulator sessions. Capture waits a full window.

    @property
    def inference_model(self) -> nn.Module:
        return self.actor

    @property
    def actor_onnx_wrapper(self) -> nn.Module:
        return copy.deepcopy(self.actor).cpu().eval()

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        target = torch.device(device or self.device)
        actor = self.actor if device is None else copy.deepcopy(self.actor).to(target).eval()

        def policy(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            with torch.no_grad():
                action: torch.Tensor = actor(obs["actor_obs"].to(target))
            return action

        return policy

    def _eval_mode(self) -> None:
        self.actor.eval()

    def _compute_eval_actions(self, actor_state: dict[str, Any]) -> dict[str, Any]:
        actor_state["actions"] = self.actor(actor_state["obs"]["actor_obs"])
        return actor_state

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None) -> None:
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state: dict[str, Any] = {"obs": self.env.reset_all()}
        try:
            for step in itertools.islice(itertools.count(), max_eval_steps):
                actor_state["step"] = step
                actor_state = self._pre_eval_env_step(actor_state)
                actor_state = self.env_step(actor_state)
                actor_state = self._post_eval_env_step(actor_state)
        finally:
            self._post_evaluate_policy()
            self.writer.close()

    def export(self, onnx_file_path: str) -> None:
        export_policy_as_onnx(
            self.actor_onnx_wrapper, onnx_file_path, {"actor_obs": torch.zeros(1, self.io.obs_scale.numel())}
        )
        robot = self.env.robot_config
        kp, kd = get_control_gains_from_config(robot)
        urdf_path, urdf = get_urdf_text_from_robot_config(robot)
        observation = self.env.observation_manager
        attach_onnx_metadata(
            onnx_path=onnx_file_path,
            metadata={
                "dof_names": robot.dof_names,
                "kp": kp,
                "kd": kd,
                "action_scale": self.io.control.action_scales.cpu().tolist(),
                "command_ranges": get_command_ranges_from_env(self.env),
                "robot_urdf": urdf,
                "robot_urdf_path": urdf_path,
                "observation_layout": observation.get_observation_layout(["actor_obs"]),
                "observation_clip": observation.cfg.clip_observations,
                **self._checkpoint_metadata(self.iteration),
            },
        )
