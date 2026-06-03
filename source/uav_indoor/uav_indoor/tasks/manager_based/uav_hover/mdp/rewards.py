from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .events import get_hover_target

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hover_height_tracking(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bounded (0, 1] reward for being at the commanded height: exp(-(z_err/std)^2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    err = get_hover_target(env) - z
    return torch.exp(-torch.square(err) / (std**2))


def horizontal_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize world-frame horizontal drift (station-keeping without position feedback)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=1)
