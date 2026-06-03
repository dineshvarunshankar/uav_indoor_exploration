from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .events import get_hover_target

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
