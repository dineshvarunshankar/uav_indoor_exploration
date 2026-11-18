from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, wrap_to_pi

import math

from rlPx4Controller.pyParallelControl import ParallelVelControl


class PX4VelocityAction(ActionTerm):
    """PX4 velocity control with SysId thrust/torque applied on rotor links (+Z up)."""

    cfg: PX4VelocityActionCfg

    def __init__(self, cfg: PX4VelocityActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=True
        )
        self._prop_body_ids, self._prop_body_names = self._asset.find_bodies(
            cfg.prop_body_names, preserve_order=True
        )
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._throttle = torch.zeros(self.num_envs, 4, device=self.device)

        self._scale = torch.tensor(
            [cfg.v_max_xy, cfg.v_max_xy, cfg.v_max_z, cfg.yaw_max],
            device=self.device,
        )
        self._motor_spin_sign = torch.tensor(cfg.motor_spin_sign, device=self.device)
        self._motor_torque_sign = torch.tensor(cfg.motor_torque_sign, device=self.device)
        self._k_thrust = abs(cfg.k_thrust)
        self._k_torque = abs(cfg.k_torque)

        self._forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self._zero_joint_effort = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

        self._px4 = ParallelVelControl(self.num_envs)

        self._thrust_state = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust_init = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._motor_tau = torch.full((self.num_envs, 1), cfg.motor_tau, device=self.device)
        self._k_thrust_scale = torch.ones(self.num_envs, 1, device=self.device)

        self._buf_len = max(int(cfg.max_action_delay_steps) + 1, 1)
        self._action_hist = torch.zeros(self.num_envs, self._buf_len, 4, device=self.device)
        self._hist_ptr = 0
        self._act_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_arange = torch.arange(self.num_envs, device=self.device)

        self._state_buf_len = max(int(cfg.ctrl_state_delay_steps) + 1, 1)
        self._state_hist = torch.zeros(self.num_envs, self._state_buf_len, 6, device=self.device)
        self._state_ptr = 0
        self._state_init = False

        self._in_air = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _throttle_to_omega(self, throttle: torch.Tensor) -> torch.Tensor:
        """Map post-THR_MDL_FAC motor signal [0, 1] to rotor speed (rad/s)."""
        rpm = self.cfg.rpm_min + throttle * (self.cfg.rpm_max - self.cfg.rpm_min)
        omega = rpm * (2.0 * math.pi / 60.0)
        return omega.clamp(self.cfg.omega_min, self.cfg.omega_max)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Once per env step: policy actions -> (latency-delayed) velocity + yaw setpoints for rlPx4."""
        self._raw_actions[:] = actions

        self._action_hist[:, self._hist_ptr] = actions
        read_idx = (self._hist_ptr - self._act_delay) % self._buf_len
        delayed = self._action_hist[self._env_arange, read_idx]
        self._hist_ptr = (self._hist_ptr + 1) % self._buf_len

        squashed = torch.tanh(delayed)
        self._processed_actions[:, :3] = squashed[:, :3] * self._scale[:3]
        _, _, yaw = euler_xyz_from_quat(self._asset.data.root_quat_w)
        self._processed_actions[:, 3] = wrap_to_pi(yaw + squashed[:, 3] * self.cfg.yaw_delta_max)

    def _delayed_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Root state as the controller sees it, optionally delayed by N physics steps."""
        data = self._asset.data
        if self._state_buf_len == 1:
            return data.root_pos_w, data.root_quat_w, data.root_lin_vel_w, data.root_ang_vel_w

        now = torch.cat([data.root_pos_w, data.root_lin_vel_w], dim=-1)
        if not self._state_init:
            self._state_hist[:] = now.unsqueeze(1)
            self._state_init = True
        self._state_hist[:, self._state_ptr] = now
        read_idx = (self._state_ptr - int(self.cfg.ctrl_state_delay_steps)) % self._state_buf_len
        delayed = self._state_hist[:, read_idx]
        self._state_ptr = (self._state_ptr + 1) % self._state_buf_len
        return delayed[:, 0:3], data.root_quat_w, delayed[:, 3:6], data.root_ang_vel_w

    def apply_actions(self) -> None:
        """Once per physics step: rlPx4 -> per-rotor thrust/torque on rotor bodies (+Z)."""
        data = self._asset.data
        dt = self._env.physics_dt

        pos_t, quat_t, vel_t, ang_vel_t = self._delayed_state()
        pos = pos_t.detach().cpu().numpy()
        vel = vel_t.detach().cpu().numpy()
        ang_vel = ang_vel_t.detach().cpu().numpy()
        quat_w = quat_t

        vel_b = self._processed_actions[:, :3]
        vel_w = quat_apply(quat_w, vel_b)
        actions_w = torch.cat([vel_w, self._processed_actions[:, 3:4]], dim=-1)
        actions_np = actions_w.detach().cpu().numpy()
        quat_np = quat_w.detach().cpu().numpy()

        z_agl = self._asset.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        self._in_air |= z_agl > self.cfg.in_air_height
        in_air_np = self._in_air.float().detach().cpu().numpy()

        self._px4.set_status(pos, quat_np, vel, ang_vel, dt, in_air_np)
        motor_np = self._px4.update(actions_np)
        motor_cmd = torch.as_tensor(motor_np, device=self.device, dtype=torch.float32)

        self._throttle[:] = motor_cmd
        self._throttle.clamp_(0.0, 1.0)
        self._throttle *= self.cfg.motor_scale
        if self.cfg.motor_clip is not None:
            lo, hi = self.cfg.motor_clip
            self._throttle.clamp_(lo, hi)

        omega = self._throttle_to_omega(self._throttle)
        omega_sq = omega.square()

        thrust_cmd = (self._k_thrust * self._k_thrust_scale) * omega_sq

        not_init = ~self._thrust_init
        if not_init.any():
            self._thrust_state[not_init] = thrust_cmd[not_init]
            self._thrust_init[not_init] = True
        alpha = dt / (self._motor_tau + dt)
        self._thrust_state += alpha * (thrust_cmd - self._thrust_state)
        thrust = self._thrust_state
        torque_mag = (self._k_torque / self._k_thrust) * thrust

        self._forces.zero_()
        self._forces[:, :, 2] = thrust
        self._torques.zero_()
        self._torques[:, :, 2] = self._motor_torque_sign * torque_mag

        self._asset.instantaneous_wrench_composer.set_forces_and_torques(
            forces=self._forces,
            torques=self._torques,
            body_ids=self._prop_body_ids,
            is_global=False,
        )

        self._asset.set_joint_effort_target(self._zero_joint_effort, joint_ids=self._joint_ids)

        if self.cfg.visual_spin_joints:
            self._asset.set_joint_velocity_target(
                self._motor_spin_sign * omega, joint_ids=self._joint_ids
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device)
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._throttle.zero_()
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._throttle[env_ids_t] = 0.0

        self._raw_actions[env_ids_t, :3] = 0.0
        self._processed_actions[env_ids_t, :3] = 0.0

        _, _, yaw = euler_xyz_from_quat(self._asset.data.root_quat_w[env_ids_t])
        yaw = wrap_to_pi(yaw)
        self._processed_actions[env_ids_t, 3] = yaw
        self._raw_actions[env_ids_t, 3] = 0.0

        self._thrust_state[env_ids_t] = 0.0
        self._thrust_init[env_ids_t] = False
        self._in_air[env_ids_t] = False

        if self._state_buf_len > 1:
            data = self._asset.data
            now = torch.cat([data.root_pos_w, data.root_lin_vel_w], dim=-1)[env_ids_t]
            self._state_hist[env_ids_t] = now.unsqueeze(1)

        n = env_ids_t.numel()
        self._action_hist[env_ids_t] = 0.0

        if self.cfg.randomize:
            tau_lo, tau_hi = self.cfg.motor_tau_range
            self._motor_tau[env_ids_t, 0] = tau_lo + (tau_hi - tau_lo) * torch.rand(n, device=self.device)
            kt_lo, kt_hi = self.cfg.k_thrust_scale_range
            self._k_thrust_scale[env_ids_t, 0] = kt_lo + (kt_hi - kt_lo) * torch.rand(n, device=self.device)
            self._act_delay[env_ids_t] = torch.randint(0, self._buf_len, (n,), device=self.device)
        else:
            self._motor_tau[env_ids_t, 0] = self.cfg.motor_tau
            self._k_thrust_scale[env_ids_t, 0] = 1.0
            self._act_delay[env_ids_t] = 0

        self._px4.reset(env_ids_t.detach().cpu().tolist())


class PX4WaypointAction(PX4VelocityAction):
    """Body-frame local waypoint action backed by the existing PX4 velocity loop."""

    cfg: PX4WaypointActionCfg

    def __init__(self, cfg: PX4WaypointActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._local_waypoint_b = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def local_waypoint_b(self) -> torch.Tensor:
        return self._local_waypoint_b

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._action_hist[:, self._hist_ptr] = actions
        read_idx = (self._hist_ptr - self._act_delay) % self._buf_len
        delayed = self._action_hist[self._env_arange, read_idx]
        self._hist_ptr = (self._hist_ptr + 1) % self._buf_len

        squashed = torch.tanh(delayed)
        scales = torch.tensor(
            [self.cfg.max_delta_xy, self.cfg.max_delta_xy, self.cfg.max_delta_z],
            device=self.device,
        )
        self._local_waypoint_b[:] = squashed[:, :3] * scales

        velocity_b = self._local_waypoint_b / self.cfg.waypoint_time_constant_s
        self._processed_actions[:, 0] = velocity_b[:, 0].clamp(-self.cfg.v_max_xy, self.cfg.v_max_xy)
        self._processed_actions[:, 1] = velocity_b[:, 1].clamp(-self.cfg.v_max_xy, self.cfg.v_max_xy)
        self._processed_actions[:, 2] = velocity_b[:, 2].clamp(-self.cfg.v_max_z, self.cfg.v_max_z)

        _, _, yaw = euler_xyz_from_quat(self._asset.data.root_quat_w)
        self._processed_actions[:, 3] = wrap_to_pi(yaw + squashed[:, 3] * self.cfg.yaw_delta_max)


@configclass
class PX4VelocityActionCfg(ActionTermCfg):
    """Velocity + yaw setpoints via rlPx4; thrust on rotor links from sim_model SysId."""

    class_type: type[ActionTerm] = PX4VelocityAction

    joint_names: list[str] = ["joint0", "joint1", "joint2", "joint3"]
    prop_body_names: list[str] = ["rotor0", "rotor1", "rotor2", "rotor3"]

    v_max_xy: float = 3.0
    v_max_z: float = 1.0
    yaw_max: float = math.pi
    yaw_delta_max: float = 0.3

    rpm_min: float = 2000.0    # VOXL_ESC_RPM_MIN
    rpm_max: float = 15000.0   # VOXL_ESC_RPM_MAX
    omega_min: float = 0.0
    omega_max: float = 1451.3110862033648   # 13859 rpm motor max

    k_thrust: float = -6.228022812483619e-07
    k_torque: float = 5.9902683625785134e-09

    motor_spin_sign: tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0)
    motor_torque_sign: tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0)

    motor_scale: float = 1.0
    motor_clip: tuple[float, float] | None = (0.0, 1.0)
    visual_spin_joints: bool = True

    motor_tau: float = 0.05

    randomize: bool = True
    motor_tau_range: tuple[float, float] = (0.03, 0.07)
    k_thrust_scale_range: tuple[float, float] = (0.90, 1.05)
    max_action_delay_steps: int = 2

    ctrl_state_delay_steps: int = 0

    in_air_height: float = 0.25

@configclass
class PX4WaypointActionCfg(PX4VelocityActionCfg):
    """Short body-frame waypoint action using the PX4 velocity controller underneath."""

    class_type: type[ActionTerm] = PX4WaypointAction

    max_delta_xy: float = 1.2
    max_delta_z: float = 0.6
    waypoint_time_constant_s: float = 0.6
