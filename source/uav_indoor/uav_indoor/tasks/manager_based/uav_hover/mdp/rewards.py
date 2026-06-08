from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from .events import get_episode_yaw_ref, get_hover_target, get_episode_xy_ref
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hover_height_tracking(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking commanded height: exp(-(z_err/std)^2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    err = get_hover_target(env) - z
    return torch.exp(-torch.square(err) / (std**2))


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
    # squared variant: torch.exp(-torch.square(yaw_err) / (std**2))
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

def base_ang_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)