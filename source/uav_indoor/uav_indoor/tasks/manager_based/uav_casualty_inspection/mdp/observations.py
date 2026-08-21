"""Deployable actor and privileged critic observations."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi


def temporal_depth(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the latest relative-disparity frame as ``(N, 1, H, W)``."""

    return env.perception.depth_history[:, -1:]


def pose_tracks(env: ManagerBasedEnv) -> torch.Tensor:
    return env.perception.tracker.observation().flatten(1)


def perception_freshness(env: ManagerBasedEnv) -> torch.Tensor:
    return env.perception.freshness


def local_reference_body(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    offset_w = env.mission_reference.position_w - asset.data.root_pos_w
    offset_b = quat_apply_inverse(asset.data.root_quat_w, offset_w)
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    yaw_error = wrap_to_pi(env.mission_reference.yaw_w - yaw)
    return torch.cat((offset_b, yaw_error[:, None]), dim=-1)


def vehicle_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    velocity_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.root_lin_vel_w)
    angular_velocity_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.root_ang_vel_w)
    projected_gravity = asset.data.projected_gravity_b
    return torch.cat((velocity_b, angular_velocity_b, projected_gravity), dim=-1)


def previous_action(env: ManagerBasedEnv) -> torch.Tensor:
    term = env.action_manager.get_term("px4_velocity")
    return term.safe_actions


def mission_phase(env: ManagerBasedEnv, phase_count: int = 7) -> torch.Tensor:
    return torch.nn.functional.one_hot(env.mission_reference.phase, num_classes=phase_count).float()


def mission_status(env: ManagerBasedEnv) -> torch.Tensor:
    """Mission state the phase one-hot cannot express. ``recording`` is omitted: it is
    exactly ``phase == RECORD`` and already carried by :func:`mission_phase`."""

    return torch.stack(
        (
            env.mission_reference.target_valid.float(),
            env.mission_reference.arc_progress,
        ),
        dim=-1,
    )


def privileged_scene_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Metric truth only for critic/reward/evaluation."""

    robot_position = env.scene["robot"].data.root_pos_w
    casualties = (env.casualty_truth_w - robot_position[:, None, :]).flatten(1)
    hazards = (env.dynamic_hazard_positions_w - robot_position[:, None, :]).flatten(1)
    reference = local_reference_body(env)
    state = vehicle_state(env)
    phase = mission_phase(env)
    return torch.cat(
        (
            casualties,
            hazards,
            env.casualty_active.float(),
            env.hazard_active.float(),
            env.minimum_clearance[:, None],
            reference,
            state,
            phase,
        ),
        dim=-1,
    )
