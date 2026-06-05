from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def get_hover_target(env: ManagerBasedEnv) -> torch.Tensor:
    """Per-env commanded hover height (world z, meters)"""
    if not hasattr(env, "hover_target_z"):
        env.hover_target_z = torch.zeros(env.num_envs, device=env.device)
    return env.hover_target_z


def reset_hover_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    height_range: tuple[float, float],
) -> None:
    """Sample a new commanded hover height per env (mode='reset'/'startup')."""
    target = get_hover_target(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device) #if none, use all envs (startup mode)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    n = len(env_ids)
    target[env_ids] = math_utils.sample_uniform(
        height_range[0], height_range[1], (n,), device=env.device
    )

def get_episode_yaw_ref(env: ManagerBasedEnv) -> torch.Tensor:
    """Per-env yaw (rad) at episode start, after reset_base."""
    if not hasattr(env, "episode_yaw_ref"):
        env.episode_yaw_ref = torch.zeros(env.num_envs, device=env.device)
    return env.episode_yaw_ref

def record_episode_yaw_ref(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    asset: Articulation = env.scene[asset_cfg.name]
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w[env_ids])
    get_episode_yaw_ref(env)[env_ids] = wrap_to_pi(yaw)