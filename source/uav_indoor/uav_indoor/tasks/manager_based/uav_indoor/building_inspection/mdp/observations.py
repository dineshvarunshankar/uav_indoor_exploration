"""Actor and privileged critic observations."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from ..critic_features import privileged_navigation_features
from ..waypoint import world_to_heading_frame
from ..policy_features import (
    disparity_freshness_features,
    disparity_image,
    motion_mode_features,
    previous_action_features,
    reference_features,
    selected_detection_features,
    state_features,
    tof_features,
)


def _robot_position(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_pos_w - env.scene.env_origins


def compact_state(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vehicle motion and attitude available from the onboard state estimator."""

    asset = env.scene[asset_cfg.name]
    state_fresh = getattr(env, "state_estimate_fresh", None)
    if state_fresh is None:
        state_fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return state_features(
        linear_velocity_b=asset.data.root_lin_vel_b,
        angular_velocity_b=asset.data.root_ang_vel_b,
        projected_gravity_b=asset.data.projected_gravity_b,
        height_above_ground_m=_robot_position(env)[:, 2],
        state_fresh=state_fresh,
    )


def compact_reference(env: ManagerBasedEnv) -> torch.Tensor:
    """Direction, range, and continuous yaw error to the supervisor waypoint."""

    asset = env.scene["robot"]
    offset_w = env.mission_reference.position_w - _robot_position(env)
    offset_b = world_to_heading_frame(offset_w, asset.data.root_quat_w)
    yaw_error = wrap_to_pi(env.mission_reference.yaw_w - env.robot_yaw_w)
    return reference_features(offset_b, yaw_error)


def motion_mode(env: ManagerBasedEnv) -> torch.Tensor:
    return motion_mode_features(env.mission_reference.phase)


def opening_detection(env: ManagerBasedEnv) -> torch.Tensor:
    env._refresh()
    return selected_detection_features(
        box_xywhn=env.active_opening_box,
        confidence=env.active_opening_confidence,
        age_s=env.active_opening_age_s,
        valid=env.active_opening_box_valid,
        target_changed=env.active_target_changed,
    )


def compact_disparity(env: ManagerBasedEnv) -> torch.Tensor:
    """Latest 64 by 64 relative-disparity image for the learned encoder."""

    return disparity_image(env.disparity.observation())


def compact_disparity_freshness(env: ManagerBasedEnv) -> torch.Tensor:
    """Disparity validity and age for the vector observation."""

    return disparity_freshness_features(env.disparity.freshness())


def memory_reset(env: ManagerBasedEnv) -> torch.Tensor:
    """Reset actor memory when the committed opening identity changes."""

    return env.active_target_changed.float()[:, None]


def compact_tof(env: ManagerBasedEnv) -> torch.Tensor:
    """Selected-opening ToF evidence with explicit UNKNOWN state."""

    env._refresh()
    return tof_features(
        range_m=env.tof_range_m,
        valid=env.tof_valid,
        clear=env.tof_clear,
        point_count=env.tof_point_count,
        threshold_configured=env.tof_threshold_configured,
        max_range_m=env.cfg.tof.max_range_m,
        required_points=env.cfg.tof.min_region_points,
    )


def previous_action(env: ManagerBasedEnv) -> torch.Tensor:
    return previous_action_features(env.action_manager.prev_action)


def privileged_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the shared critic features."""

    robot = _robot_position(env)
    heading = torch.stack(
        (torch.cos(env.robot_yaw_w), torch.sin(env.robot_yaw_w), torch.zeros_like(env.robot_yaw_w)),
        dim=-1,
    )
    return privileged_navigation_features(
        target_offset_w=env.mission_reference.position_w - robot,
        clearance_direction_w=heading,
        clearance_m=env.clearance_m,
        clearance_known=env.clearance_known,
    )
