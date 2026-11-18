"""Task-local bounded velocity action with PX4 dynamics and optional depth shield."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from rlPx4Controller.pyParallelControl import ParallelVelControl

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, wrap_to_pi

from ..control import VelocityLimits, bounded_velocity_step


class PX4VelocityAction(ActionTerm):
    """Map policy output to bounded body velocity and yaw-rate commands."""

    cfg: PX4VelocityActionCfg

    def __init__(self, cfg: PX4VelocityActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._prop_body_ids, _ = self._asset.find_bodies(cfg.prop_body_names, preserve_order=True)

        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._safe_actions = torch.zeros_like(self._raw_actions)
        self._yaw_setpoint = torch.zeros(self.num_envs, device=self.device)
        self._shield_intervened = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._shield_total = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
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
    def shield_intervened(self) -> torch.Tensor:
        return self._shield_intervened

    @property
    def shield_total(self) -> torch.Tensor:
        return self._shield_total

    def _apply_shield(self, actions: torch.Tensor) -> torch.Tensor:
        """Clamp commands that fly into something."""

        safe = actions.clone()
        self._shield_intervened.zero_()
        if not self.cfg.shield_enabled:
            return safe

        perception = self._env.perception
        depth = perception.depth_history[:, -1]
        height, width = depth.shape[-2:]
        centre = depth[:, height // 3 : 2 * height // 3, width // 3 : 2 * width // 3]

        centre_near = torch.quantile(centre.flatten(1), self.cfg.shield_depth_quantile, dim=1)
        frame_near = depth.flatten(1).median(dim=1).values
        obstacle = (centre_near - frame_near) > self.cfg.shield_near_margin

        person = perception.nearest_track_range_m < self.cfg.shield_min_person_range_m

        forward = safe[:, 0] > 0.0
        blocked = (obstacle | person) & forward
        safe[:, 0] = torch.where(blocked, torch.zeros_like(safe[:, 0]), safe[:, 0])

        open_side = torch.where(
            depth[:, :, : width // 2].mean(dim=(1, 2)) < depth[:, :, width // 2 :].mean(dim=(1, 2)),
            -1.0,
            1.0,
        )
        safe[:, 1] += torch.where(blocked, open_side * self.cfg.shield_lateral_speed, torch.zeros_like(open_side))

        agl = self._asset.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        sinking = (agl < self.cfg.shield_min_height_m) & (safe[:, 2] < 0.0)
        safe[:, 2] = torch.where(sinking, torch.zeros_like(safe[:, 2]), safe[:, 2])

        self._shield_intervened[:] = blocked | sinking
        self._shield_total += self._shield_intervened.long()
        return safe

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
        self._safe_actions[:] = self._apply_shield(self._processed_actions)
        _, _, measured_yaw = euler_xyz_from_quat(self._asset.data.root_quat_w)
        self._yaw_setpoint[:] = wrap_to_pi(measured_yaw + self._safe_actions[:, 3] * dt)

    def _throttle_to_omega(self, throttle: torch.Tensor) -> torch.Tensor:
        rpm = self.cfg.rpm_min + throttle * (self.cfg.rpm_max - self.cfg.rpm_min)
        return (rpm * (2.0 * math.pi / 60.0)).clamp(self.cfg.omega_min, self.cfg.omega_max)

    def apply_actions(self) -> None:
        data = self._asset.data
        quaternion = data.root_quat_w
        height = data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        self._in_air |= height > self.cfg.in_air_height

        velocity_w = quat_apply(quaternion, self._safe_actions[:, :3])
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
        motor_command = torch.as_tensor(
            self._px4.update(np.ascontiguousarray(host[:, 14:18])),
            device=self.device,
            dtype=torch.float32,
        )
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
        self._forces[:, :, 2] = self._thrust_state
        self._torques.zero_()
        self._torques[:, :, 2] = self._motor_torque_sign * torque
        self._asset.instantaneous_wrench_composer.set_forces_and_torques(
            forces=self._forces,
            torques=self._torques,
            body_ids=self._prop_body_ids,
            is_global=False,
        )
        self._asset.set_joint_effort_target(self._zero_joint_effort, joint_ids=self._joint_ids)
        if self.cfg.visual_spin_joints:
            self._asset.set_joint_velocity_target(self._motor_spin_sign * omega, joint_ids=self._joint_ids)

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
        self._shield_intervened[env_ids_t] = False
        self._shield_total[env_ids_t] = 0
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
    accel_max_xy: float = 3.0
    accel_max_z: float = 3.0
    yaw_accel_max: float = 1.2

    rpm_min: float = 2000.0
    rpm_max: float = 15000.0
    omega_min: float = 0.0
    omega_max: float = 1451.3110862033648
    k_thrust: float = -6.228022812483619e-07
    k_torque: float = 5.9902683625785134e-09
    motor_spin_sign: tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0)
    motor_torque_sign: tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0)
    motor_scale: float = 1.0
    motor_tau: float = 0.02
    visual_spin_joints: bool = True
    in_air_height: float = 0.25

    randomize: bool = False
    motor_tau_range: tuple[float, float] = (0.03, 0.07)
    k_thrust_scale_range: tuple[float, float] = (0.90, 1.05)
    max_action_delay_steps: int = 0

    shield_enabled: bool = True
    shield_depth_quantile: float = 0.80
    shield_near_margin: float = 0.25
    shield_min_person_range_m: float = 1.2
    shield_min_height_m: float = 0.15
    shield_lateral_speed: float = 0.35
