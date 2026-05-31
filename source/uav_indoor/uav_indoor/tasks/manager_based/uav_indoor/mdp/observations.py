from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from .opening_targets import get_opening_targets
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_offset_body(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    delta_b = offset_to_opening_body(env, asset)
    dist = torch.norm(delta_b, dim=-1, keepdim=True)
    direction = delta_b / (dist + 1e-8)
    return torch.cat([direction, dist], dim=-1)