"""PX4 velocity and position actions for building inspection."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from rlPx4Controller.pyParallelControl import ParallelPosControl, ParallelVelControl

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, wrap_to_pi

from ..control import VelocityLimits, bounded_velocity_step
from ..policy_features import motion_mode_features
from ..waypoint import (
    body_relative_position_target,
    heading_yaw_w,
    update_stale_hold,
)

OBSERVE_MOTION_MODE = 3


class PX4VelocityAction(ActionTerm):
    """Map policy output to bounded body velocity and yaw-rate commands."""

    cfg: PX4VelocityActionCfg

    def __init__(self, cfg: PX4VelocityActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._prop_body_ids, _ = self._asset.find_bodies(cfg.prop_body_names, preserve_order=True)
        airframe_ids, _ = self._asset.find_bodies([cfg.airframe_body_name], preserve_order=True)
        self._wrench_body_ids = list(self._prop_body_ids) + list(airframe_ids)

        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._safe_actions = torch.zeros_like(self._raw_actions)
        self._yaw_setpoint = torch.zeros(self.num_envs, device=self.device)
        self._action_saturated_total = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._forces = torch.zeros(self.num_envs, len(self._wrench_body_ids), 3, device=self.device)
        self._torques = torch.zeros_like(self._forces)
        self._zero_joint_effort = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)
        self._throttle = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust_state = torch.zeros_like(self._throttle)
        self._thrust_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._motor_tau = torch.full((self.num_envs, 1), cfg.motor_tau, device=self.device)
        self._thrust_scale = torch.ones(self.num_envs, 1, device=self.device)
        self._in_air = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._motor_spin_sign = torch.tensor(cfg.motor_spin_sign, device=self.device)
        self._motor_torque_sign = torch.tensor(cfg.motor_torque_sign, device=self.device)
        self._px4 = ParallelVelControl(self.num_envs)

        self._delay_length = max(cfg.max_action_delay_steps + 1, 1)
        self._action_history = torch.zeros(self.num_envs, self._delay_length, 4, device=self.device)
        self._delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._history_index = 0
        self._env_indices = torch.arange(self.num_envs, device=self.device)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def safe_actions(self) -> torch.Tensor:
        return self._safe_actions

    @property
    def action_saturated_total(self) -> torch.Tensor:
        return self._action_saturated_total

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._action_history[:, self._history_index] = actions
        read_index = (self._history_index - self._delay_steps) % self._delay_length
        delayed = self._action_history[self._env_indices, read_index]
        self._history_index = (self._history_index + 1) % self._delay_length

        dt = self._env.step_dt
        self._processed_actions[:] = bounded_velocity_step(
            delayed,
            self._processed_actions,
            dt,
            VelocityLimits(
                v_max_xy=self.cfg.v_max_xy,
                v_max_z=self.cfg.v_max_z,
                yaw_rate_max=self.cfg.yaw_rate_max,
                accel_max_xy=self.cfg.accel_max_xy,
                accel_max_z=self.cfg.accel_max_z,
                yaw_accel_max=self.cfg.yaw_accel_max,
            ),
        )
        self._safe_actions[:] = self._processed_actions
        self._action_saturated_total += (self._raw_actions.abs() >= 0.99).any(dim=1).long()
        _, _, measured_yaw = euler_xyz_from_quat(self._asset.data.root_quat_w)
        lead = (self._safe_actions[:, 3] / self.cfg.yaw_p_gain).clamp(
            -self.cfg.yaw_lead_max, self.cfg.yaw_lead_max
        )
        self._yaw_setpoint[:] = wrap_to_pi(measured_yaw + lead)

    def _throttle_to_omega(self, throttle: torch.Tensor) -> torch.Tensor:
        rpm = self.cfg.rpm_min + throttle * (self.cfg.rpm_max - self.cfg.rpm_min)
        return (rpm * (2.0 * math.pi / 60.0)).clamp(self.cfg.omega_min, self.cfg.omega_max)

    def _apply_velocity_setpoint(self, velocity_w: torch.Tensor) -> None:
        """Run the existing PX4 and motor path for a world-frame velocity setpoint."""

        data = self._asset.data
        quaternion = data.root_quat_w
        height = data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        self._in_air |= height > self.cfg.in_air_height

        setpoint = torch.cat((velocity_w, self._yaw_setpoint[:, None]), dim=-1)
        host = torch.cat(
            (
                data.root_pos_w,
                quaternion,
                data.root_lin_vel_w,
                data.root_ang_vel_w,
                self._in_air.float()[:, None],
                setpoint,
            ),
            dim=-1,
        ).detach().cpu().numpy()
        self._px4.set_status(
            np.ascontiguousarray(host[:, 0:3]),
            np.ascontiguousarray(host[:, 3:7]),
            np.ascontiguousarray(host[:, 7:10]),
            np.ascontiguousarray(host[:, 10:13]),
            self._env.physics_dt,
            np.ascontiguousarray(host[:, 13]),
        )
        self._drive_motors(
            torch.as_tensor(
                self._px4.update(np.ascontiguousarray(host[:, 14:18])),
                device=self.device,
                dtype=torch.float32,
            )
        )

    def _drive_motors(self, motor_command: torch.Tensor) -> None:
        """Motor command to rotor thrust and airframe reaction couple."""

        self._throttle[:] = motor_command.clamp(0.0, 1.0) * self.cfg.motor_scale
        omega = self._throttle_to_omega(self._throttle)
        thrust_command = abs(self.cfg.k_thrust) * self._thrust_scale * omega.square()
        initialize = ~self._thrust_initialized
        self._thrust_state[initialize] = thrust_command[initialize]
        self._thrust_initialized[initialize] = True
        alpha = self._env.physics_dt / (self._motor_tau + self._env.physics_dt)
        self._thrust_state += alpha * (thrust_command - self._thrust_state)
        torque = (abs(self.cfg.k_torque) / abs(self.cfg.k_thrust)) * self._thrust_state

        self._forces.zero_()
        self._forces[:, :-1, 2] = self._thrust_state
        self._torques.zero_()
        self._torques[:, -1, 2] = (self._motor_torque_sign * torque).sum(dim=-1)
        self._asset.instantaneous_wrench_composer.set_forces_and_torques(
            forces=self._forces,
            torques=self._torques,
            body_ids=self._wrench_body_ids,
            is_global=False,
        )
        self._asset.set_joint_effort_target(self._zero_joint_effort, joint_ids=self._joint_ids)
        if self.cfg.visual_spin_joints:
            self._asset.set_joint_velocity_target(self._motor_spin_sign * omega, joint_ids=self._joint_ids)

    def apply_actions(self) -> None:
        velocity_w = quat_apply(self._asset.data.root_quat_w, self._safe_actions[:, :3])
        self._apply_velocity_setpoint(velocity_w)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        _, _, yaw = euler_xyz_from_quat(self._asset.data.root_quat_w[env_ids_t])
        self._yaw_setpoint[env_ids_t] = wrap_to_pi(yaw)
        self._raw_actions[env_ids_t] = 0.0
        self._processed_actions[env_ids_t] = 0.0
        self._safe_actions[env_ids_t] = 0.0
        self._throttle[env_ids_t] = 0.0
        self._thrust_state[env_ids_t] = 0.0
        self._thrust_initialized[env_ids_t] = False
        self._in_air[env_ids_t] = False
        self._action_saturated_total[env_ids_t] = 0
        self._action_history[env_ids_t] = 0.0

        count = env_ids_t.numel()
        if self.cfg.randomize:
            low, high = self.cfg.motor_tau_range
            self._motor_tau[env_ids_t, 0] = low + (high - low) * torch.rand(count, device=self.device)
            low, high = self.cfg.k_thrust_scale_range
            self._thrust_scale[env_ids_t, 0] = low + (high - low) * torch.rand(count, device=self.device)
            self._delay_steps[env_ids_t] = torch.randint(0, self._delay_length, (count,), device=self.device)
        else:
            self._motor_tau[env_ids_t, 0] = self.cfg.motor_tau
            self._thrust_scale[env_ids_t, 0] = 1.0
            self._delay_steps[env_ids_t] = 0
        self._px4.reset(env_ids_t.detach().cpu().tolist())


@configclass
class PX4VelocityActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PX4VelocityAction

    joint_names: list[str] = ["joint0", "joint1", "joint2", "joint3"]
    prop_body_names: list[str] = ["rotor0", "rotor1", "rotor2", "rotor3"]
    v_max_xy: float = 3.0
    v_max_z: float = 1.0
    yaw_rate_max: float = math.radians(45.0)  # MPC_YAWRAUTO_MAX
    yaw_p_gain: float = 2.8  # MC_YAW_P
    yaw_lead_max: float = 0.5
    accel_max_xy: float = 3.0
    accel_max_z: float = 3.0
    yaw_accel_max: float = 1.2

    rpm_min: float = 2000.0
    rpm_max: float = 15000.0
    omega_min: float = 0.0
    omega_max: float = 1451.3110862033648
    k_thrust: float = -6.228022812483619e-07
    k_torque: float = 0.05 * 6.228022812483619e-07
    motor_spin_sign: tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0)
    airframe_body_name: str = "body"
    motor_torque_sign: tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0)
    motor_scale: float = 1.0
    motor_tau: float = 0.02
    visual_spin_joints: bool = True
    in_air_height: float = 0.25

    randomize: bool = False
    motor_tau_range: tuple[float, float] = (0.03, 0.07)
    k_thrust_scale_range: tuple[float, float] = (0.90, 1.05)
    max_action_delay_steps: int = 0




class PX4PositionAction(PX4VelocityAction):
    """Map a body-relative policy action to a short PX4 position waypoint."""

    cfg: PX4PositionActionCfg

    def __init__(self, cfg: PX4PositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._px4 = ParallelPosControl(self.num_envs)
        self._target_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._stale_hold_position_w = torch.zeros_like(self._target_w)
        self._stale_hold_yaw_w = torch.zeros(self.num_envs, device=self.device)
        self._stale_hold_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._reach_m = torch.tensor(
            (cfg.v_max_xy, cfg.v_max_xy, cfg.v_max_z), device=self.device
        ) * cfg.reach_seconds
        self._yaw_reach_rad = cfg.yaw_rate_max * cfg.reach_seconds

    @property
    def target_w(self) -> torch.Tensor:
        return self._target_w

    def _apply_position_setpoint(self, target_w: torch.Tensor) -> None:
        """Apply an environment-local PX4 position setpoint."""

        data = self._asset.data
        local = data.root_pos_w - self._env.scene.env_origins
        self._in_air |= local[:, 2] > self.cfg.in_air_height
        setpoint = torch.cat((target_w, self._yaw_setpoint[:, None]), dim=-1)
        host = torch.cat(
            (
                local,
                data.root_quat_w,
                data.root_lin_vel_w,
                data.root_ang_vel_w,
                self._in_air.float()[:, None],
                setpoint,
            ),
            dim=-1,
        ).detach().cpu().numpy()
        self._px4.set_status(
            np.ascontiguousarray(host[:, 0:3]),
            np.ascontiguousarray(host[:, 3:7]),
            np.ascontiguousarray(host[:, 7:10]),
            np.ascontiguousarray(host[:, 10:13]),
            self._env.physics_dt,
            np.ascontiguousarray(host[:, 13]),
        )
        self._drive_motors(
            torch.as_tensor(
                self._px4.update(np.ascontiguousarray(host[:, 14:18])),
                device=self.device,
                dtype=torch.float32,
            )
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        clipped = actions.clamp(-1.0, 1.0)
        self._raw_actions[:] = clipped
        self._action_history[:, self._history_index] = clipped
        read_index = (self._history_index - self._delay_steps) % self._delay_length
        delayed = self._action_history[self._env_indices, read_index]
        self._history_index = (self._history_index + 1) % self._delay_length
        self._processed_actions[:] = delayed

        data = self._asset.data
        position_w = data.root_pos_w - self._env.scene.env_origins
        yaw_w = heading_yaw_w(data.root_quat_w)
        reference = self._env.mission_reference
        mode = motion_mode_features(reference.phase).argmax(dim=-1)
        translation_allowed = mode != OBSERVE_MOTION_MODE

        state = torch.cat(
            (position_w, data.root_quat_w, data.root_lin_vel_w, data.root_ang_vel_w),
            dim=-1,
        )
        state_finite = torch.isfinite(state).all(dim=-1)
        state_fresh = getattr(self._env, "state_estimate_fresh", None)
        if state_fresh is None:
            state_fresh = state_finite
        else:
            state_fresh = (
                torch.as_tensor(state_fresh, device=self.device, dtype=torch.bool)
                & state_finite
            )
        disparity_fresh = self._env.disparity.freshness()[:, 0].to(dtype=torch.bool)
        input_fresh = state_fresh & disparity_fresh
        (
            self._stale_hold_position_w,
            self._stale_hold_yaw_w,
            self._stale_hold_active,
        ) = update_stale_hold(
            position_w=position_w,
            yaw_w=yaw_w,
            input_fresh=input_fresh,
            held_position_w=self._stale_hold_position_w,
            held_yaw_w=self._stale_hold_yaw_w,
            hold_active=self._stale_hold_active,
        )

        target_w, target_yaw, safe_action = body_relative_position_target(
            position_w,
            yaw_w,
            delayed,
            self._reach_m,
            self._yaw_reach_rad,
            translation_allowed,
            reference.position_w,
            input_fresh,
            self._stale_hold_position_w,
            self._stale_hold_yaw_w,
        )
        self._target_w[:] = target_w
        self._yaw_setpoint[:] = target_yaw
        self._safe_actions[:] = safe_action
        self._action_saturated_total += (delayed.abs() >= 0.99).any(dim=1).long()

    def apply_actions(self) -> None:
        self._apply_position_setpoint(self._target_w)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._target_w[ids] = (
            self._asset.data.root_pos_w[ids] - self._env.scene.env_origins[ids]
        )
        self._stale_hold_position_w[ids] = self._target_w[ids]
        self._stale_hold_yaw_w[ids] = heading_yaw_w(
            self._asset.data.root_quat_w[ids]
        )
        self._stale_hold_active[ids] = False


@configclass
class PX4PositionActionCfg(PX4VelocityActionCfg):
    class_type: type[ActionTerm] = PX4PositionAction

    reach_seconds: float = 0.5
