from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat

from .events import get_hover_target
from .events import get_episode_xy_ref

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

# height error
def hover_height_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Commanded height minus current height (world z, meters), shape (num_envs, 1).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    target = get_hover_target(env)
    return (target - z).unsqueeze(-1)
# xy error
def hover_xy_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Target xy minus current xy, in the body (yaw) frame (m), shape (num_envs, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    err_w = get_episode_xy_ref(env) - xy
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    c, s = torch.cos(yaw), torch.sin(yaw)
    err_bx = c * err_w[:, 0] + s * err_w[:, 1]
    err_by = -s * err_w[:, 0] + c * err_w[:, 1]
    return torch.stack([err_bx, err_by], dim=-1)