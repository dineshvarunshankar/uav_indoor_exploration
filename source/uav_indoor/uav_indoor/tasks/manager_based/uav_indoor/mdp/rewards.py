# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Penalize joint position deviation from a target value."""
#     # extract the used quantities (to enable type-hinting)
#     asset: Articulation = env.scene[asset_cfg.name]
#     # wrap the joint positions to (-pi, pi)
#     joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
#     # compute the reward
#     return torch.sum(torch.square(joint_pos - target), dim=1)


def _get_asset(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]

def _robot_pos_env(env:ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    return asset.data.root_pos_w - env.scene.env_origins

def _dist_to_opening(env:ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    if not hasattr(env, "opening_target_env"):
        return torch.zeros(env.num_envs, device=env.device)
    return torch.norm(env.opening_target_env - _robot_pos_env(env, asset), dim=-1)


#task rewards

def distance_to_opening_exp(
    env:ManagerBasedRLEnv,
    std: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset = _get_asset(env, asset_cfg)
    dist = _dist_to_opening(env, asset)
    return torch.exp(-torch.square(dist) / (std**2))


def progress_to_opening(
    env:ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = _get_asset(env, asset_cfg)
    dist = _dist_to_opening(env, asset)
    if not hasattr(env, "_prev_dist_to_opening"):
        env._prev_dist_to_opening = dist.clone()
        return  torch.zeros_like(dist)
    progress = env._prev_dist_to_opening - dist
    env._prev_dist_to_opening = dist.clone()
    return progress

def heading_to_opening_exp(
    env:ManagerBasedRLEnv,
    std: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset = _get_asset(env, asset_cfg)
    if not hasattr(env, "opening_target_env"):
        return torch.zeros(env.num_envs, device=env.device)
    delta = env.opening_target_env - _robot_pos_env(env, asset)
    target_yaw = torch.atan2(delta[:,1], delta[:,0])
    _,_, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw = wrap_to_pi(yaw)
    yaw_err = math_utils.wrap_to_pi(target_yaw - yaw)
    return torch.exp(-torch.square(yaw_err) / (std**2))

def body_axis_aligned_to_opening(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward forward body axis pointing at opening (3D). Assumes body +X is forward."""
    asset = _get_asset(env, asset_cfg)
    if not hasattr(env, "opening_target_env"):
        return torch.zeros(env.num_envs, device=env.device)
    delta_w = env.opening_target_env - _robot_pos_env(env, asset)
    delta_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, delta_w)
    delta_b = delta_b / (torch.norm(delta_b, dim=-1, keepdim=True) + 1e-8)
    forward_b = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    return torch.clamp(torch.sum(delta_b * forward_b, dim=-1), min=0.0, max=1.0)

def at_opening(
    env: ManagerBasedRLEnv,
    success_radius: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = _get_asset(env, asset_cfg)
    d = _dist_to_opening(env, asset)
    return (d < success_radius).float()

def height_error_to_opening_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = _get_asset(env, asset_cfg)
    if not hasattr(env, "opening_target_env"):
        return torch.zeros(env.num_envs, device=env.device)
    z_err = _robot_pos_env(env, asset)[:, 2] - env.opening_target_env[:, 2]
    return torch.square(z_err)