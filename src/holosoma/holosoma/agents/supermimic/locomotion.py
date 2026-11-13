"""Translate the G1 locomotion task into tensors for differentiable rollouts."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

from holosoma.envs.locomotion.locomotion_manager import LeggedRobotLocomotionManager
from holosoma.managers.observation.terms import locomotion as obs_terms
from holosoma.managers.reward.terms import locomotion as reward_terms
from holosoma.utils.rotations import quat_apply
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.managers.action.terms.joint_control import JointPositionActionTerm
    from holosoma.managers.command.terms.locomotion import LocomotionGait
    from holosoma.managers.terrain.terms.locomotion import TerrainLocomotion


class SuperMimicLocomotion(LeggedRobotLocomotionManager):
    action_delay_idx: torch.Tensor
    model_io: LocomotionModelIO | None = None
    model_successor: tuple[torch.Tensor, torch.Tensor] | None = None
    model_push: tuple[torch.Tensor, torch.Tensor] | None = None
    model_root_successor: torch.Tensor | None = None

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.model_push = None
        super()._pre_physics_step(actions)

    def _push_robots(self, env_ids: torch.Tensor) -> None:
        if self.model_io is not None:
            self.model_push = (env_ids, obs_terms.get_base_lin_vel(self)[env_ids])
        super()._push_robots(env_ids)

    def _check_termination(self) -> None:
        super()._check_termination()
        if self.model_io is not None:
            # Heights and physical state have been refreshed, and resets have not run yet.
            successor = self.model_io.state()
            if self.model_push is not None:
                env_ids, velocity = self.model_push
                successor[env_ids, :3] = velocity * self.model_io.state_scale[:3]
            self.model_successor = (successor, self.model_io.applied_actions().clone())
            self.model_root_successor = self.model_io.root_pose()


class LocomotionState(NamedTuple):
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    gravity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    foot_heights: torch.Tensor
    foot_separation: torch.Tensor
    foot_orientation_penalty: torch.Tensor
    terrain: torch.Tensor


class LocomotionModelIO:
    """Share the task's control and observation conventions with imagined G1 worlds."""

    def __init__(self, env: SuperMimicLocomotion):
        self.env = env
        self.is_isaacgym = env.simulator.simulator_config.name == "isaacgym"
        if not self.is_isaacgym:
            raise ValueError("This experiment requires Isaac Gym Preview 4")
        self.context_encoding = "isaacgym_physical_v1"
        self.control = cast("JointPositionActionTerm", env.action_manager.get_term("joint_control"))
        self.gait = cast("LocomotionGait", env.command_manager.get_state("locomotion_gait"))
        self.terrain = cast("TerrainLocomotion", env.terrain_manager.get_state("locomotion_terrain"))
        if self.control.action_queue is not None and self.control.action_queue.shape[1] > 2:
            raise ValueError("SuperMimic supports at most one control step of action delay.")

        robot = env.robot_config
        nominal = env.default_dof_pos_base.squeeze(0)
        lower = nominal.new_tensor(robot.dof_pos_lower_limit_list)
        upper = nominal.new_tensor(robot.dof_pos_upper_limit_list)
        self.action_bound = torch.maximum(nominal - lower, upper - nominal) / robot.control.action_scale
        self.nominal_height = robot.init_state.pos[2]
        self.state_sizes = (3, 3, 3, env.num_dof, env.num_dof, 2, 1, 1, self._ground_heights().shape[1])
        self.state_scale = torch.cat(
            [
                nominal.new_full((size,), scale)
                for size, scale in zip(self.state_sizes, (2.0, 0.25, 1.0, 1.0, 0.05, 5.0, 5.0, 1.0, 5.0))
            ]
        )
        self.static_context = self._read_physics_context()

        manager = env.observation_manager
        group = manager.cfg.groups["actor_obs"]
        self.actor_terms = sorted(group.terms)
        self.term_slices = {}
        if not group.concatenate or group.history_length != 1:
            raise ValueError("SuperMimic requires the concatenated single-frame G1 actor observation.")
        prefix = "holosoma.managers.observation.terms.locomotion:"
        supported = {
            "actions",
            "base_ang_vel",
            "base_lin_vel",
            "command_ang_vel",
            "command_lin_vel",
            "cos_phase",
            "dof_pos",
            "dof_vel",
            "projected_gravity",
            "sin_phase",
        }
        scale, noise, lower_bound, upper_bound = [], [], [], []
        action_slice = None
        offset = 0
        for name in self.actor_terms:
            term = group.terms[name]
            if name not in supported or term.func != prefix + name or term.params:
                raise ValueError(f"Unsupported SuperMimic actor observation term: {name}")
            dim = manager._compute_term("actor_obs", name, term).shape[-1]
            if name == "actions":
                action_slice = slice(offset, offset + dim)
            self.term_slices[name] = slice(offset, offset + dim)
            offset += dim
            scale.append(nominal.new_full((dim,), term.scale))
            noise.append(nominal.new_full((dim,), term.noise if group.enable_noise else 0.0))
            low, high = term.clip if term.clip is not None else (-float("inf"), float("inf"))
            if getattr(term, "global_clip", True):
                limit = manager.cfg.clip_observations
                low = min(max(low, -limit), limit)
                high = min(max(high, -limit), limit)
            lower_bound.append(nominal.new_full((dim,), low))
            upper_bound.append(nominal.new_full((dim,), high))
        self.obs_scale = torch.cat(scale)
        self.obs_noise = torch.cat(noise)
        self.obs_min = torch.cat(lower_bound)
        self.obs_max = torch.cat(upper_bound)
        self.obs_action_scale = torch.ones_like(self.obs_scale)
        if action_slice is not None:
            self.obs_action_scale[action_slice] = self.action_bound.reciprocal()

    def _ground_heights(self) -> torch.Tensor:
        return self.terrain._ray_hits_world_base[..., 2]

    def root_pose(self) -> torch.Tensor:
        # Isaac Gym and Holosoma expose xyzw; integration uses explicit wxyz.
        root = self.env.simulator.robot_root_states
        return torch.cat((root[:, :3], root[:, 6:7], root[:, 3:6]), -1).clone()

    def applied_actions(self) -> torch.Tensor:
        return self.control._actions_after_delay

    def _read_physics_context(self) -> torch.Tensor:
        return self._read_isaacgym_context()

    def set_context_encoding(self, encoding: str) -> None:
        """Restore the checkpoint's feature convention without reinterpreting weights."""
        allowed = {"isaacgym_physical_v1", "isaacgym_batch_centered_v0"}
        if encoding not in allowed:
            raise ValueError(f"Unsupported physics-context encoding for this simulator: {encoding}")
        self.context_encoding = encoding
        self.static_context = self._read_physics_context()

    def _read_isaacgym_context(self) -> torch.Tensor:
        sim = self.env.simulator
        torso_idx = self.env.torso_index
        masses, centers, frictions = [], [], []
        for env_ptr, actor in zip(sim.envs, sim.robot_handles):
            bodies = sim.gym.get_actor_rigid_body_properties(env_ptr, actor)
            shapes = sim.gym.get_actor_rigid_shape_properties(env_ptr, actor)
            com = bodies[torso_idx].com
            masses.append([body.mass for body in bodies])
            centers.append([com.x, com.y, com.z])
            frictions.append([shapes[0].friction])
        nominal = self.env.default_dof_pos_base
        mass = nominal.new_tensor(masses).clamp_min(1e-6).log()
        com = nominal.new_tensor(centers)
        if self.context_encoding == "isaacgym_batch_centered_v0":
            # Format-3 checkpoints omitted the training reference. Preserve their
            # legacy behavior explicitly; do not pretend to recover that reference.
            mass, com = mass - mass.mean(0), com - com.mean(0)
        # log(mass / 1 kg) and COM / 0.1 m have a fixed physical reference.
        # Unrelated worlds, world count and evaluation seeds cannot move it.
        return torch.cat((mass, com * 10, nominal.new_tensor(frictions)), dim=-1)

    def context(self) -> torch.Tensor:
        kp, kd = self.control.get_pd_scale_tensors()
        n = self.env.num_envs
        return torch.cat((self.static_context.expand(n, -1), kp.expand(n, -1) - 1, kd.expand(n, -1) - 1), dim=1)

    def state(self) -> torch.Tensor:
        env = self.env
        height = env.simulator.robot_root_states[:, 2:3] - self._ground_heights() - self.nominal_height
        height = height.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        feet = self.terrain.feet_heights.nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)
        return (
            torch.cat(
                (
                    obs_terms.get_base_lin_vel(env),
                    obs_terms.get_base_ang_vel(env),
                    obs_terms.get_projected_gravity(env),
                    obs_terms.dof_pos(env),
                    obs_terms.dof_vel(env),
                    feet,
                    self._lateral_feet_distance()[:, None],
                    reward_terms.penalty_feet_ori(env)[:, None],
                    height,
                ),
                dim=-1,
            )
            * self.state_scale
        )

    def _lateral_feet_distance(self) -> torch.Tensor:
        env = self.env
        feet = env.simulator._rigid_body_pos[:, env.feet_indices, :2]
        delta = feet[:, 0] - feet[:, 1]
        forward = quat_apply(env.base_quat, reward_terms.base_forward_vector(env), w_last=True)
        yaw = torch.atan2(forward[:, 1], forward[:, 0])
        return (yaw.cos() * delta[:, 1] - yaw.sin() * delta[:, 0]).abs()

    def unpack(self, state: torch.Tensor) -> LocomotionState:
        return LocomotionState(*(state / self.state_scale).split(self.state_sizes, dim=-1))

    def actor_observation(
        self,
        state: LocomotionState,
        command: torch.Tensor,
        phase: torch.Tensor,
        previous_action: torch.Tensor,
        *,
        with_noise: bool = True,
    ) -> torch.Tensor:
        terms = {
            "actions": previous_action,
            "base_ang_vel": state.angular_velocity,
            "base_lin_vel": state.linear_velocity,
            "command_ang_vel": command[:, 2:3],
            "command_lin_vel": command[:, :2],
            "cos_phase": phase.cos(),
            "dof_pos": state.joint_position,
            "dof_vel": state.joint_velocity,
            "projected_gravity": state.gravity,
            "sin_phase": phase.sin(),
        }
        obs = torch.cat([terms[name] for name in self.actor_terms], dim=-1)
        if with_noise:
            obs = obs + (torch.rand_like(obs) * 2 - 1) * self.obs_noise
        return (obs * self.obs_scale).clamp(self.obs_min, self.obs_max)

    def delay(self) -> torch.Tensor:
        if self.control.action_queue is None:
            return torch.zeros((self.env.num_envs, 1), device=self.env.device, dtype=torch.bool)
        return self.env.action_delay_idx[:, None].bool()

    def applied_action(
        self, action: torch.Tensor, previous_action: torch.Tensor, delayed: torch.Tensor
    ) -> torch.Tensor:
        action = torch.where(delayed, previous_action, action)
        control = self.env.robot_config.control
        if control.clip_actions:
            action = action.clamp(-control.action_clip_value, control.action_clip_value)
        return action / self.action_bound

    def reward(self) -> LocomotionReward:
        return LocomotionReward(self.env)


class LocomotionReward:
    """The G1 reward computed on predicted successors using the live curriculum weights."""

    def __init__(self, env: SuperMimicLocomotion):
        manager = env.reward_manager
        self.dt = env.dt
        self.only_positive = manager.cfg.only_positive_rewards
        self.terms = {name: manager.get_term_cfg(name) for name in manager.active_terms}
        prefix = "holosoma.managers.reward.terms.locomotion:"
        supported = {
            "tracking_lin_vel",
            "tracking_ang_vel",
            "penalty_ang_vel_xy",
            "penalty_orientation",
            "penalty_action_rate",
            "feet_phase",
            "pose",
            "penalty_close_feet_xy",
            "penalty_feet_ori",
            "alive",
        }
        for name, cfg in self.terms.items():
            if name not in supported or cfg.func != prefix + name:
                raise ValueError(f"Unsupported SuperMimic reward term: {name}")
        self.pose_weights = (
            torch.tensor(self.terms["pose"].params["pose_weights"], device=env.device) if "pose" in self.terms else None
        )

    def __call__(
        self,
        state: LocomotionState,
        command: torch.Tensor,
        phase: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        reward = torch.zeros_like(command[:, 0])
        for name, cfg in self.terms.items():
            if cfg.weight == 0:
                continue
            params = cfg.params
            if name == "tracking_lin_vel":
                error = (command[:, :2] - state.linear_velocity[:, :2]).square().sum(dim=-1)
                value = torch.exp(-error / params.get("tracking_sigma", 0.25))
            elif name == "tracking_ang_vel":
                error = (command[:, 2] - state.angular_velocity[:, 2]).square()
                value = torch.exp(-error / params.get("tracking_sigma", 0.25))
            elif name == "penalty_ang_vel_xy":
                value = state.angular_velocity[:, :2].square().sum(dim=-1)
            elif name == "penalty_orientation":
                value = state.gravity[:, :2].square().sum(dim=-1)
            elif name == "penalty_action_rate":
                value = (action - previous_action).square().sum(dim=-1)
            elif name == "feet_phase":
                expected = reward_terms._expected_foot_height(phase, params.get("swing_height", 0.08))
                value = torch.exp(-(state.foot_heights - expected).square().sum(-1) / params.get("tracking_sigma", 0.25))
            elif name == "pose":
                assert self.pose_weights is not None
                value = (state.joint_position.square() * self.pose_weights).sum(dim=-1)
            elif name == "penalty_close_feet_xy":
                value = (state.foot_separation[:, 0] < params.get("close_feet_threshold", 0.05)).float()
            elif name == "penalty_feet_ori":
                value = state.foot_orientation_penalty[:, 0].clamp_min(0)
            else:  # alive
                value = torch.ones_like(reward)
            reward = reward + cfg.weight * self.dt * value
        return reward.clamp_min(0) if self.only_positive else reward
