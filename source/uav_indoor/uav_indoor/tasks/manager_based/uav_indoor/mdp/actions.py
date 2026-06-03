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
        self._throttle_poly = torch.tensor(
            [
                cfg.throttle_omega_c0,
                cfg.throttle_omega_c1,
                cfg.throttle_omega_c2,
                cfg.throttle_omega_c3,
            ],
            device=self.device,
        )
        self._motor_spin_sign = torch.tensor(cfg.motor_spin_sign, device=self.device)
        self._motor_torque_sign = torch.tensor(cfg.motor_torque_sign, device=self.device)
        self._k_thrust = abs(cfg.k_thrust)
        self._k_torque = abs(cfg.k_torque)

        self._forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self._zero_joint_effort = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

        # rlPx4 runs on CPU/numpy
        self._px4 = ParallelVelControl(self.num_envs)

        # --- actuator first-order lag + per-env domain-randomization state ---
        self._thrust_state = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust_init = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._motor_tau = torch.full((self.num_envs, 1), cfg.motor_tau, device=self.device)
        self._k_thrust_scale = torch.ones(self.num_envs, 1, device=self.device)

        # --- action-latency ring buffer (units: env steps) ---
        self._buf_len = max(int(cfg.max_action_delay_steps) + 1, 1)
        self._action_hist = torch.zeros(self.num_envs, self._buf_len, 4, device=self.device)
        self._hist_ptr = 0
        self._act_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_arange = torch.arange(self.num_envs, device=self.device)

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
        """Map normalized motor throttle [0, 1] to rotor speed (rad/s)."""
        c0, c1, c2, c3 = self._throttle_poly
        omega = (c0 * throttle**3) + (c1 * throttle**2) + (c2 * throttle) + c3
        return omega.clamp(self.cfg.omega_min, self.cfg.omega_max)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Once per env step: policy actions -> (latency-delayed) velocity + yaw setpoints for rlPx4."""
        self._raw_actions[:] = actions

        # inject per-env command latency via a ring buffer of env steps
        self._action_hist[:, self._hist_ptr] = actions
        read_idx = (self._hist_ptr - self._act_delay) % self._buf_len
        delayed = self._action_hist[self._env_arange, read_idx]
        self._hist_ptr = (self._hist_ptr + 1) % self._buf_len

        self._processed_actions[:] = delayed * self._scale

    def apply_actions(self) -> None:
        """Once per physics step: rlPx4 -> per-rotor thrust/torque on rotor bodies (+Z)."""
        data = self._asset.data
        dt = self._env.physics_dt

        pos = data.root_pos_w.detach().cpu().numpy()
        vel = data.root_lin_vel_w.detach().cpu().numpy()
        ang_vel = data.root_ang_vel_w.detach().cpu().numpy()
        quat_w = data.root_quat_w

        vel_b = self._processed_actions[:, :3]
        vel_w = quat_apply(quat_w, vel_b)
        actions_w = torch.cat([vel_w, self._processed_actions[:, 3:4]], dim=-1)
        actions_np = actions_w.detach().cpu().numpy()
        quat_np = quat_w.detach().cpu().numpy()

        self._px4.set_status(pos, quat_np, vel, ang_vel, dt)
        motor_np = self._px4.update(actions_np)

        self._throttle[:] = torch.as_tensor(motor_np, device=self.device, dtype=torch.float32)
        self._throttle.clamp_(0.0, 1.0)
        self._throttle *= self.cfg.motor_scale
        if self.cfg.motor_clip is not None:
            lo, hi = self.cfg.motor_clip
            self._throttle.clamp_(lo, hi)

        omega = self._throttle_to_omega(self._throttle)
        omega_sq = omega.square()

        # static thrust target; per-env k_thrust scale models battery/motor variation
        thrust_cmd = (self._k_thrust * self._k_thrust_scale) * omega_sq

        # first-order actuator lag (per-env tau from step-response SysID); seed to
        # the first command on the step after reset to avoid a spurious startup dip
        not_init = ~self._thrust_init
        if not_init.any():
            self._thrust_state[not_init] = thrust_cmd[not_init]
            self._thrust_init[not_init] = True
        alpha = dt / (self._motor_tau + dt)
        self._thrust_state += alpha * (thrust_cmd - self._thrust_state)
        thrust = self._thrust_state
        # reaction torque shares the same (lagged) omega^2 dependence
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

        # Rotor joints do not provide lift via effort; only external wrench does.
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
        self._raw_actions[env_ids_t, 3] = yaw / self.cfg.yaw_max

        # re-initialize actuator-lag state (re-seeds to first command next step)
        self._thrust_state[env_ids_t] = 0.0
        self._thrust_init[env_ids_t] = False

        # seed the action-latency buffer with the reset command (vel=0, yaw held)
        n = env_ids_t.numel()
        self._action_hist[env_ids_t] = 0.0
        self._action_hist[env_ids_t, :, 3] = (yaw / self.cfg.yaw_max).unsqueeze(1)

        # per-env domain randomization, resampled each reset
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


@configclass
class PX4VelocityActionCfg(ActionTermCfg):
    """Velocity + yaw setpoints via rlPx4; thrust on rotor links from sim_model SysId."""

    class_type: type[ActionTerm] = PX4VelocityAction

    joint_names: list[str] = ["joint0", "joint1", "joint2", "joint3"]
    prop_body_names: list[str] = ["rotor0", "rotor1", "rotor2", "rotor3"]

    v_max_xy: float = 3.0
    v_max_z: float = 1.0
    yaw_max: float = math.pi

    # Throttle -> omega ( throttle_to_omega_rads)
    throttle_omega_c0: float = 248.004161
    throttle_omega_c1: float = -1198.228360
    throttle_omega_c2: float = 2445.931020
    throttle_omega_c3: float = -41.786093
    omega_min: float = 0.0
    omega_max: float = 2500.0

    # from SysID
    k_thrust: float = 6.228023e-07
    k_torque: float = 5.990268e-09

    # Spin direction for joint_vel visualization (+1 CCW, -1 CW when viewed from +Z)
    motor_spin_sign: tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0)
    # Reaction torque on rotor link (+Z); opposite of spin for CCW pair
    motor_torque_sign: tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0)

    motor_scale: float = 1.0
    motor_clip: tuple[float, float] | None = (0.0, 1.0)
    visual_spin_joints: bool = True

    # --- actuator dynamics ---
    motor_tau: float = 0.161  # first-order thrust lag (s), from step-response SysID

    # --- domain randomization (per-env, resampled each reset) ---
    randomize: bool = True
    motor_tau_range: tuple[float, float] = (0.12, 0.20)      # spin-up time constant spread
    k_thrust_scale_range: tuple[float, float] = (0.90, 1.05)  # battery sag / motor variation
    max_action_delay_steps: int = 2                           # command latency, in env steps
