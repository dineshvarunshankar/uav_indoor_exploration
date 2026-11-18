"""Reset and curriculum events for the inspection mission."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def reset_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_range: tuple[float, float] = (-8.0, 8.0),
    height_range: tuple[float, float] = (1.5, 3.0),
) -> None:
    """Reset the aircraft inside a configurable mission-local spawn square."""

    asset: Articulation = env.scene[asset_cfg.name]
    count = len(env_ids)
    low, high = xy_range
    position = torch.empty(count, 3, device=env.device)
    position[:, 0] = math_utils.sample_uniform(low, high, (count,), env.device)
    position[:, 1] = math_utils.sample_uniform(low, high, (count,), env.device)
    position[:, 2] = math_utils.sample_uniform(height_range[0], height_range[1], (count,), env.device)
    position += env.scene.env_origins[env_ids]
    roll = math_utils.sample_uniform(-0.05, 0.05, (count,), env.device)
    pitch = math_utils.sample_uniform(-0.05, 0.05, (count,), env.device)
    yaw = math_utils.sample_uniform(-torch.pi, torch.pi, (count,), env.device)
    orientation = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    velocity = torch.zeros(count, 6, device=env.device)
    velocity[:, :3] = math_utils.sample_uniform(-0.1, 0.1, (count, 3), env.device)
    asset.write_root_pose_to_sim(torch.cat((position, orientation), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocity, env_ids=env_ids)


def reset_mission(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Resample casualties, hazards, GPS error, and mission state."""

    env.reset_episode(env_ids)


def apply_wind(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Apply per-axis wind force and torque."""

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    size = (len(env_ids), num_bodies)
    forces = torch.stack(
        (
            math_utils.sample_uniform(*force_range["x"], size, asset.device),
            math_utils.sample_uniform(*force_range["y"], size, asset.device),
            math_utils.sample_uniform(*force_range["z"], size, asset.device),
        ),
        dim=-1,
    )
    torques = torch.stack(
        (
            math_utils.sample_uniform(*torque_range["x"], size, asset.device),
            math_utils.sample_uniform(*torque_range["y"], size, asset.device),
            math_utils.sample_uniform(*torque_range["z"], size, asset.device),
        ),
        dim=-1,
    )
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces,
        torques=torques,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
        is_global=True,
    )
