"""Manager terms for random-goal obstacle navigation."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

from ..building_inspection.critic_features import privileged_navigation_features
from ..building_inspection.policy_features import (
    disparity_freshness_features,
    disparity_image,
    motion_mode_features,
    reference_features,
    state_features,
)
from ..building_inspection.reward_features import (
    altitude_violation as altitude_violation_feature,
)
from ..building_inspection.reward_features import (
    body_rate_cost,
    collision_features,
    yaw_acceleration_cost,
)
from ..building_inspection.waypoint import world_to_heading_frame
from .action_observation import latest_policy_action
from .contact_history import control_step_maximum_contact_force
from .geometry import nearest_geometry
from .policy import navigation_policy_features
from .reward_features import (
    altitude_margin_cost,
    bounded_stopping_risk,
    clipped_action_change,
    depthnav_acceleration_cost,
    depthnav_collision_cost,
    depthnav_jerk_cost,
    depthnav_yaw_alignment_reward,
    physical_clearance,
    safe_goal_event_rate,
    update_episode_safety,
)
from .reward_features import goal_progress as goal_progress_feature
from .reward_features import workspace_violation as workspace_violation_feature


def randomize_dome_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    color_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("dome_light"),
) -> None:
    """Change the shared dome light without changing geometry or collision shapes."""

    del env_ids
    intensity_low, intensity_high = intensity_range
    color_low, color_high = color_range
    if intensity_low <= 0.0 or intensity_high < intensity_low:
        raise ValueError("invalid dome-light intensity range")
    if color_low < 0.0 or color_high > 1.0 or color_high < color_low:
        raise ValueError("invalid dome-light color range")
    light = env.scene[asset_cfg.name].prims[0]
    intensity = intensity_low + (intensity_high - intensity_low) * torch.rand((), device=env.device)
    color = color_low + (color_high - color_low) * torch.rand(3, device=env.device)
    light.GetAttribute("inputs:intensity").Set(float(intensity.item()))
    light.GetAttribute("inputs:color").Set(tuple(float(value) for value in color.tolist()))


def _position(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_pos_w - env.scene.env_origins


def _goal_delta(env: ManagerBasedEnv) -> torch.Tensor:
    return env.goal_w - _position(env)


def _max_contact_force(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return control_step_maximum_contact_force(
        sensor.data.net_forces_w_history,
        env.cfg.decimation,
    )


def _nearest_geometry(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return nearest_geometry(
        _position(env),
        env.obstacle_centres_w,
        env.obstacle_sizes_m,
        env.obstacle_active,
        env.obstacle_cylinder_slots,
    )


def policy_observation(env: ManagerBasedEnv) -> torch.Tensor:
    asset = env.scene["robot"]
    fresh = env.state_estimate_fresh
    state = state_features(
        asset.data.root_lin_vel_b,
        asset.data.root_ang_vel_b,
        asset.data.projected_gravity_b,
        _position(env)[:, 2],
        fresh,
    )
    delta_b = world_to_heading_frame(_goal_delta(env), asset.data.root_quat_w)
    yaw_error = wrap_to_pi(env.mission_reference.yaw_w - env.robot_yaw_w)
    reference = reference_features(delta_b, yaw_error)
    mode = motion_mode_features(env.mission_reference.phase)
    freshness = disparity_freshness_features(env.disparity.freshness())
    previous = latest_policy_action(env.action_manager, "px4_waypoint")
    return navigation_policy_features(state, reference, mode, freshness, previous)


def disparity_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """Latest 64 by 64 relative-disparity image for the shared actor encoder."""

    return disparity_image(env.disparity.observation())


def memory_reset_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """Stage one changes no targets inside an episode."""

    return torch.zeros((env.num_envs, 1), device=env.device)


def privileged_observation(env: ManagerBasedEnv) -> torch.Tensor:
    distance, direction, known = _nearest_geometry(env)
    return privileged_navigation_features(
        target_offset_w=_goal_delta(env),
        clearance_direction_w=direction,
        clearance_m=distance,
        clearance_known=known,
    )


def collision_event(env: ManagerBasedEnv, threshold: float = 1.0) -> torch.Tensor:
    event, _ = collision_features(_max_contact_force(env), threshold)
    return event / env.step_dt


def collision_severity(env: ManagerBasedEnv, threshold: float = 1.0) -> torch.Tensor:
    _, severity = collision_features(_max_contact_force(env), threshold)
    return severity


def goal_progress(env: ManagerBasedEnv) -> torch.Tensor:
    distance = torch.linalg.vector_norm(_goal_delta(env), dim=-1)
    progress, env._reward_previous_distance, env._reward_distance_valid = goal_progress_feature(
        env._reward_previous_distance,
        distance,
        env._reward_distance_valid,
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        env.step_dt,
    )
    return progress


def goal_reached(env: ManagerBasedEnv) -> torch.Tensor:
    position = _position(env)
    obstacle_distance, _, obstacle_known = _nearest_geometry(env)
    clearance = physical_clearance(
        obstacle_distance,
        obstacle_known,
        position[:, 2],
        env.cfg.vehicle_radius_m,
    )
    env._reward_episode_safe = update_episode_safety(
        env._reward_episode_safe,
        clearance,
        position[:, 2],
        env.cfg.training_clearance_margin_m,
        env.cfg.minimum_altitude_m,
    )
    reached = torch.linalg.vector_norm(_goal_delta(env), dim=-1) <= env.cfg.goal_tolerance_m
    event, env._reward_goal_latched = safe_goal_event_rate(
        reached,
        collision(env),
        out_of_bounds(env),
        env._reward_episode_safe,
        env._reward_goal_latched,
        env.step_dt,
    )
    return event


def workspace_event(env: ManagerBasedEnv) -> torch.Tensor:
    violation = workspace_violation_feature(
        _position(env),
        env.bounds_min_m,
        env.bounds_max_m,
        clearance_m=env.cfg.vehicle_radius_m,
    )
    return violation.float() / env.step_dt



def collision_avoidance(env: ManagerBasedEnv) -> torch.Tensor:
    """Adapt DepthNav's collision loss to physical obstacle surfaces."""

    distance, direction, known = _nearest_geometry(env)
    return depthnav_collision_cost(
        velocity_w=env.scene["robot"].data.root_lin_vel_w,
        obstacle_direction_w=direction,
        obstacle_surface_distance_m=distance,
        obstacle_known=known,
        vehicle_radius_m=env.cfg.vehicle_radius_m,
        falloff_distance_m=env.cfg.danger_influence_m,
        barrier_scale=env.cfg.depthnav_barrier_scale,
        barrier_slope=env.cfg.depthnav_barrier_slope,
    )


def stopping_risk(env: ManagerBasedEnv) -> torch.Tensor:
    """Return privileged stopping risk as a training-only label."""

    distance, direction, known = _nearest_geometry(env)
    cost = depthnav_collision_cost(
        velocity_w=env.scene["robot"].data.root_lin_vel_w,
        obstacle_direction_w=direction,
        obstacle_surface_distance_m=distance,
        obstacle_known=known,
        vehicle_radius_m=env.cfg.vehicle_radius_m,
        falloff_distance_m=env.cfg.danger_influence_m,
        barrier_scale=env.cfg.depthnav_barrier_scale,
        barrier_slope=env.cfg.depthnav_barrier_slope,
    )
    return bounded_stopping_risk(cost)[:, None]

def altitude_margin(env: ManagerBasedEnv) -> torch.Tensor:
    return altitude_margin_cost(
        _position(env)[:, 2],
        env.cfg.minimum_altitude_m,
        env.cfg.altitude_margin_height_m,
    )


def altitude_violation(env: ManagerBasedEnv) -> torch.Tensor:
    return altitude_violation_feature(_position(env)[:, 2], env.cfg.minimum_altitude_m)


def action_change(env: ManagerBasedEnv) -> torch.Tensor:
    term = env.action_manager.get_term("px4_waypoint")
    return clipped_action_change(term.raw_actions, env.action_manager.prev_action)


def body_rate(env: ManagerBasedEnv) -> torch.Tensor:
    return body_rate_cost(env.scene["robot"].data.root_ang_vel_b)


def yaw_acceleration(env: ManagerBasedEnv) -> torch.Tensor:
    current = env.scene["robot"].data.root_ang_vel_b[:, 2]
    cost = yaw_acceleration_cost(current, env._reward_previous_yaw_rate, env.step_dt)
    cost = torch.where(env._reward_yaw_rate_valid, cost, torch.zeros_like(cost))
    env._reward_previous_yaw_rate = current.clone()
    env._reward_yaw_rate_valid[:] = True
    return cost


def yaw_alignment(env: ManagerBasedEnv) -> torch.Tensor:
    """Adapt DepthNav's cosine yaw reward to the current mission reference."""

    error = wrap_to_pi(env.mission_reference.yaw_w - env.robot_yaw_w).abs()
    return depthnav_yaw_alignment_reward(
        yaw_error_rad=error,
        target_distance_m=torch.linalg.vector_norm(_goal_delta(env), dim=-1),
        disable_within_m=env.cfg.depthnav_yaw_disable_within_m,
    )


def _physical_acceleration_w(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.body_lin_acc_w[:, 0, :]


def physical_acceleration(env: ManagerBasedEnv) -> torch.Tensor:
    return depthnav_acceleration_cost(_physical_acceleration_w(env))


def physical_jerk(env: ManagerBasedEnv) -> torch.Tensor:
    current = _physical_acceleration_w(env)
    cost = depthnav_jerk_cost(current, env._reward_previous_acceleration_w, env.step_dt)
    cost = torch.where(env._reward_acceleration_valid, cost, torch.zeros_like(cost))
    env._reward_previous_acceleration_w = current.clone()
    env._reward_acceleration_valid[:] = True
    return cost


def collision(env: ManagerBasedEnv, threshold: float = 1.0) -> torch.Tensor:
    event, _ = collision_features(_max_contact_force(env), threshold)
    return event.bool()


def out_of_bounds(env: ManagerBasedEnv) -> torch.Tensor:
    return workspace_violation_feature(
        _position(env),
        env.bounds_min_m,
        env.bounds_max_m,
        clearance_m=env.cfg.vehicle_radius_m,
    )


def reached_goal(env: ManagerBasedEnv) -> torch.Tensor:
    return torch.linalg.vector_norm(_goal_delta(env), dim=-1) <= env.cfg.goal_tolerance_m
