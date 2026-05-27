from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_offset_body(
    env:ManagerBasedRLEnv,
    asset_cfg:SceneEntityCfg =SceneEntityCfg(name="robot"),
) -> torch.Tensor:
    asset:Articulation = env.scene[asset_cfg.name]

    #robot pose
    robot_pos_env = asset.data.root_pos_w - env.scene.env_origins
    delta_env = env.opening_target_env - robot_pos_env
    delta_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, delta_env)
    dist = torch.norm(delta_b, dim=-1, keepdim=True)
    direction = delta_b / (dist + 1e-8)
    return torch.cat([direction, dist], dim=-1)