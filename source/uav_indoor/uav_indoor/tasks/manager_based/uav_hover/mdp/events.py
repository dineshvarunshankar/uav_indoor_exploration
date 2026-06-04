from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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
    height_range: tuple[float, float] = (1.0, 3.0),
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
