from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat, quat_apply_inverse, wrap_to_pi

from .events import get_episode_xy_ref
from .events import get_episode_yaw_ref
from .events import get_hover_target

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def base_rotation_6d(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Continuous 6D rotation: first two columns of the body->world matrix, shape (num_envs, 6)."""
    asset: Articulation = env.scene[asset_cfg.name]
    R = matrix_from_quat(asset.data.root_quat_w)
    return torch.cat([R[:, :, 0], R[:, :, 1]], dim=-1)

def hover_height_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Commanded height minus current height (world z, meters), shape (num_envs, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    target = get_hover_target(env)
    return (target - z).unsqueeze(-1)
def hover_xy_error_yaw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Target xy minus current xy, in the yaw frame (m), shape (num_envs, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    err_w = get_episode_xy_ref(env) - xy
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    c, s = torch.cos(yaw), torch.sin(yaw)
    err_bx = c * err_w[:, 0] + s * err_w[:, 1]
    err_by = -s * err_w[:, 0] + c * err_w[:, 1]
    return torch.stack([err_bx, err_by], dim=-1)


def base_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Current position relative to spawn, in the body frame (x, y, z), meters."""
    asset: Articulation = env.scene[asset_cfg.name]
    p = asset.data.root_pos_w[:, :3] - env.scene.env_origins
    return quat_apply_inverse(asset.data.root_quat_w, p)




def hover_x_error_yaw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw-frame x component of horizontal position error to target (m)."""
    return hover_xy_error_yaw(env, asset_cfg)[:, 0:1]


def hover_y_error_yaw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw-frame y component of horizontal position error to target (m)."""
    return hover_xy_error_yaw(env, asset_cfg)[:, 1:2]


def hover_yaw_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Episode heading minus current yaw, wrapped to [-pi, pi] (rad), shape (num_envs, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw_now = euler_xyz_from_quat(asset.data.root_quat_w)
    yaw_now = wrap_to_pi(yaw_now)
    yaw_err = wrap_to_pi(get_episode_yaw_ref(env) - yaw_now)
    return yaw_err.unsqueeze(-1)