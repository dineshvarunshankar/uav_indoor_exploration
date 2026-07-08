from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp import illegal_contact
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from .events import get_airborne_flag, get_episode_xy_ref, get_episode_yaw_ref, get_hover_target

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import hover_xy_error_yaw


def _height_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return env-frame height z, signed height error (target - z), and world vz."""
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    z_err = get_hover_target(env) - z
    vz = asset.data.root_lin_vel_w[:, 2]
    return z, z_err, vz


def hover_height_tracking(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Hold at commanded height: exp(-(z_err/std)^2)."""
    _, z_err, _ = _height_state(env, asset_cfg)
    return torch.exp(-torch.square(z_err) / (std**2))


def hover_height_climb(
    env: ManagerBasedRLEnv,
    height_tol: float = 0.25,
    v_cap: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward capped upward speed when clearly below target (climb phase only)."""
    _, z_err, vz = _height_state(env, asset_cfg)
    need_climb = z_err > height_tol
    climb = torch.clamp(vz, min=0.0, max=v_cap)
    return torch.where(need_climb, climb, torch.zeros_like(climb))


def hover_height_overshoot(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty magnitude when above commanded height (use negative weight)."""
    _, z_err, _ = _height_state(env, asset_cfg)
    return torch.clamp(-z_err, min=0.0)


def horizontal_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize world-frame horizontal drift."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=1)


def episode_yaw_tracking(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward holding episode heading: exp(-|yaw_err|/std)."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw_now = euler_xyz_from_quat(asset.data.root_quat_w)
    yaw_now = wrap_to_pi(yaw_now)
    yaw_err = wrap_to_pi(get_episode_yaw_ref(env) - yaw_now)
    return torch.exp(-torch.abs(yaw_err) / std)


def episode_xy_tracking(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward holding spawn xy: exp(-||xy_err||^2 / std^2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    xy_err = get_episode_xy_ref(env) - xy
    return torch.exp(-torch.sum(torch.square(xy_err), dim=-1) / (std**2))


def hold_x(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Hold the spawn x: exp(-|x_err|/std), in [0, 1]."""
    asset: Articulation = env.scene[asset_cfg.name]
    x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    return torch.exp(-torch.abs(get_episode_xy_ref(env)[:, 0] - x) / std)


def hold_y(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Hold the spawn y: exp(-|y_err|/std), in [0, 1]."""
    asset: Articulation = env.scene[asset_cfg.name]
    y = asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return torch.exp(-torch.abs(get_episode_xy_ref(env)[:, 1] - y) / std)


def hold_z(
    env: ManagerBasedRLEnv,
    std: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Hold the commanded height: exp(-|z_err|/std), in [0, 1]."""
    _, z_err, _ = _height_state(env, asset_cfg)
    return torch.exp(-torch.abs(z_err) / std)


def base_ang_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)


# --- revamped hover rewards (asymmetric height RBF, climb, damping, yaw-frame xy) ---


def hover_height_asymmetric_rbf(
    env: ManagerBasedRLEnv,
    below_std: float = 0.8,
    above_std: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Asymmetric height tracking: tolerant below target, sharper above target.

    e = target_z - z (positive when below target).
    """
    _, z_err, _ = _height_state(env, asset_cfg)
    below = z_err >= 0.0
    below_reward = torch.exp(-torch.square(z_err / below_std))
    above_reward = torch.exp(-torch.square(z_err / above_std))
    return torch.where(below, below_reward, above_reward)


def hover_height_climb_when_below(
    env: ManagerBasedRLEnv,
    height_tol: float = 0.25,
    v_cap: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward capped upward world vz when clearly below target height."""
    _, z_err, vz = _height_state(env, asset_cfg)
    need_climb = z_err > height_tol
    climb = torch.clamp(vz, min=0.0, max=v_cap)
    return torch.where(need_climb, climb, torch.zeros_like(climb))


def hover_vertical_damping_near_target(
    env: ManagerBasedRLEnv,
    near_tol: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty magnitude vz^2 when near target height (use negative weight)."""
    _, z_err, vz = _height_state(env, asset_cfg)
    near = torch.abs(z_err) < near_tol
    damping = torch.square(vz)
    return torch.where(near, damping, torch.zeros_like(damping))


def hold_x_yaw_rbf(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw-frame x position error RBF: exp(-(x_err/std)^2)."""
    x_err = hover_xy_error_yaw(env, asset_cfg)[:, 0]
    return torch.exp(-torch.square(x_err / std))


def hold_y_yaw_rbf(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw-frame y position error RBF: exp(-(y_err/std)^2)."""
    y_err = hover_xy_error_yaw(env, asset_cfg)[:, 1]
    return torch.exp(-torch.square(y_err / std))


def illegal_contact_takeoff_aware(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    min_flight_height: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Crash penalty gated by height: ground contact before first takeoff is ignored.

    Latches per-env once z > min_flight_height so a crash-to-ground later in the
    episode still counts, even though the contact itself happens back near z = 0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    airborne_flag = get_airborne_flag(env)
    airborne_flag |= z > min_flight_height
    crashed = illegal_contact(env, threshold, sensor_cfg)
    return crashed.float() * airborne_flag.float()


def crashed_after_takeoff(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    min_flight_height: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Termination: contact after the env has been airborne (bool tensor).

    Shares the airborne latch with illegal_contact_takeoff_aware so the crash
    penalty and crash termination always agree; pre-takeoff ground contact
    never terminates.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    airborne_flag = get_airborne_flag(env)
    airborne_flag |= z > min_flight_height
    return illegal_contact(env, threshold, sensor_cfg) & airborne_flag
