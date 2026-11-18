"""Pure reward features for random-goal obstacle navigation."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def stopping_influence_distance(
    braking_distance_m: float,
    speed_m_s: float,
    perception_period_s: float,
    control_period_s: float,
    vehicle_radius_m: float,
    margin_m: float,
) -> float:
    """Return the obstacle influence range from explicit physical components."""

    values = (
        braking_distance_m,
        speed_m_s,
        perception_period_s,
        control_period_s,
        vehicle_radius_m,
        margin_m,
    )
    if any(value < 0.0 for value in values):
        raise ValueError("stopping-envelope inputs must be non-negative")
    return (
        braking_distance_m
        + speed_m_s * (perception_period_s + control_period_s)
        + vehicle_radius_m
        + margin_m
    )


def stopping_clearance_influence_distance(
    braking_distance_m: float,
    speed_m_s: float,
    perception_period_s: float,
    control_period_s: float,
    margin_m: float,
) -> float:
    """Return the required body clearance before braking starts."""

    return stopping_influence_distance(
        braking_distance_m,
        speed_m_s,
        perception_period_s,
        control_period_s,
        vehicle_radius_m=0.0,
        margin_m=margin_m,
    )


def workspace_violation(
    position_m: torch.Tensor,
    bounds_min_m: torch.Tensor,
    bounds_max_m: torch.Tensor,
    clearance_m: float = 0.0,
) -> torch.Tensor:
    """Return true when a vehicle origin leaves its assigned workspace."""

    if position_m.shape != bounds_min_m.shape or position_m.shape != bounds_max_m.shape:
        raise ValueError("workspace tensors must have matching shapes")
    if clearance_m < 0.0:
        raise ValueError("clearance_m must be non-negative")
    return (
        (position_m < bounds_min_m + clearance_m)
        | (position_m > bounds_max_m - clearance_m)
    ).any(dim=-1)


def physical_clearance(
    obstacle_distance_m: torch.Tensor,
    obstacle_known: torch.Tensor,
    height_above_ground_m: torch.Tensor,
    vehicle_radius_m: float,
) -> torch.Tensor:
    """Return clearance from physical obstacles and the flat ground."""

    if vehicle_radius_m < 0.0:
        raise ValueError("vehicle_radius_m must be non-negative")
    obstacle_clearance = obstacle_distance_m - vehicle_radius_m
    obstacle_clearance = torch.where(
        obstacle_known,
        obstacle_clearance,
        torch.full_like(obstacle_clearance, torch.inf),
    )
    ground_clearance = height_above_ground_m - vehicle_radius_m
    return torch.minimum(obstacle_clearance, ground_clearance)


def _potential_progress(
    previous: torch.Tensor,
    current: torch.Tensor,
    history_valid: torch.Tensor,
    target_changed: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    progress = (previous - current) / dt
    reset = ~history_valid | target_changed
    progress = torch.where(reset, torch.zeros_like(progress), progress)
    return progress, current.clone(), torch.ones_like(history_valid)


def goal_progress(
    previous_distance_m: torch.Tensor,
    current_distance_m: torch.Tensor,
    history_valid: torch.Tensor,
    target_changed: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signed goal-distance reduction per second."""

    return _potential_progress(
        previous_distance_m, current_distance_m, history_valid, target_changed, dt
    )


def yaw_progress(
    previous_error_rad: torch.Tensor,
    current_error_rad: torch.Tensor,
    history_valid: torch.Tensor,
    target_changed: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signed reduction in absolute yaw error per second."""

    return _potential_progress(
        previous_error_rad.abs(), current_error_rad.abs(), history_valid, target_changed, dt
    )


def goal_event_rate(
    reached: torch.Tensor, previously_reached: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a one-off goal event rate compatible with Isaac Lab integration."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    event = reached & ~previously_reached
    return event.to(torch.float32) / dt, previously_reached | reached


def successful_episode(
    reached: torch.Tensor,
    collision: torch.Tensor,
    workspace_exit: torch.Tensor,
    episode_safe: torch.Tensor,
) -> torch.Tensor:
    """Return goal completion with all episode safety conditions satisfied."""

    if not (
        reached.shape
        == collision.shape
        == workspace_exit.shape
        == episode_safe.shape
    ):
        raise ValueError("episode result tensors must have matching shapes")
    return reached & ~collision & ~workspace_exit & episode_safe


def safe_goal_event_rate(
    reached: torch.Tensor,
    collision: torch.Tensor,
    workspace_exit: torch.Tensor,
    episode_safe: torch.Tensor,
    previously_reached: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pay and latch goal completion only when the terminal state is safe."""

    safe_reached = successful_episode(reached, collision, workspace_exit, episode_safe)
    return goal_event_rate(safe_reached, previously_reached, dt)


def update_episode_safety(
    previously_safe: torch.Tensor,
    clearance_m: torch.Tensor,
    altitude_m: torch.Tensor,
    minimum_clearance_m: float,
    minimum_altitude_m: float,
) -> torch.Tensor:
    """Latch any clearance or altitude violation for the complete episode."""

    if minimum_clearance_m < 0.0 or minimum_altitude_m < 0.0:
        raise ValueError("safety limits must be non-negative")
    return (
        previously_safe
        & (clearance_m >= minimum_clearance_m)
        & (altitude_m >= minimum_altitude_m)
    )


def clipped_action_change(
    current_action: torch.Tensor, previous_action: torch.Tensor
) -> torch.Tensor:
    """Return action change after both manager actions use the task's [-1, 1] domain."""

    return (current_action.clamp(-1.0, 1.0) - previous_action.clamp(-1.0, 1.0)).square().sum(dim=-1)


def blended_yaw_direction(
    goal_direction_w: torch.Tensor,
    horizontal_velocity_w: torch.Tensor,
    motion_speed_m_s: torch.Tensor,
    full_motion_weight_speed_m_s: float,
) -> torch.Tensor:
    """Blend goal bearing into travel direction as horizontal speed rises."""

    if full_motion_weight_speed_m_s <= 0.0:
        raise ValueError("full_motion_weight_speed_m_s must be positive")
    fallback = torch.zeros_like(goal_direction_w)
    fallback[..., 0] = 1.0
    goal_norm = torch.linalg.vector_norm(goal_direction_w, dim=-1, keepdim=True)
    velocity_norm = torch.linalg.vector_norm(horizontal_velocity_w, dim=-1, keepdim=True)
    goal = torch.where(
        goal_norm > 1.0e-8,
        goal_direction_w / goal_norm.clamp_min(1.0e-8),
        fallback,
    )
    motion = torch.where(
        velocity_norm > 1.0e-8,
        horizontal_velocity_w / velocity_norm.clamp_min(1.0e-8),
        goal,
    )
    weight = (motion_speed_m_s / full_motion_weight_speed_m_s).clamp(0.0, 1.0)[..., None]
    blended = (1.0 - weight) * goal + weight * motion
    blended_norm = torch.linalg.vector_norm(blended, dim=-1, keepdim=True)
    direction = functional.normalize(blended, dim=-1, eps=1.0e-8)
    return torch.where(blended_norm > 1.0e-8, direction, goal)


def clearance_margin_cost(
    distance_m: torch.Tensor,
    known: torch.Tensor,
    influence_distance_m: float,
) -> torch.Tensor:
    """Return squared proximity inside the declared obstacle influence range."""

    if influence_distance_m <= 0.0:
        raise ValueError("influence_distance_m must be positive")
    proximity = ((influence_distance_m - distance_m) / influence_distance_m).clamp(0.0, 1.0)
    return proximity.square() * known.to(proximity.dtype)


def bounded_stopping_risk(cost: torch.Tensor) -> torch.Tensor:
    """Map a non-negative stopping cost to a finite unit-interval target."""

    if torch.any(cost < 0.0):
        raise ValueError("stopping cost must be non-negative")
    finite = torch.nan_to_num(cost, nan=0.0, posinf=torch.finfo(cost.dtype).max)
    return finite / (1.0 + finite)

def depthnav_collision_cost(
    velocity_w: torch.Tensor,
    obstacle_direction_w: torch.Tensor,
    obstacle_surface_distance_m: torch.Tensor,
    obstacle_known: torch.Tensor,
    vehicle_radius_m: float,
    falloff_distance_m: float,
    barrier_scale: float = 2.5,
    barrier_slope: float = -6.0,
) -> torch.Tensor:
    """Return DepthNav's closing-speed falloff and soft intersection barrier."""

    if velocity_w.shape != obstacle_direction_w.shape:
        raise ValueError("velocity and obstacle direction must have matching shapes")
    if obstacle_surface_distance_m.shape != obstacle_known.shape:
        raise ValueError("distance and known masks must have matching shapes")
    if vehicle_radius_m < 0.0 or falloff_distance_m <= 0.0:
        raise ValueError("vehicle radius must be non-negative and falloff must be positive")
    direction = functional.normalize(obstacle_direction_w, dim=-1, eps=1.0e-8)
    closing_speed = (velocity_w * direction).sum(dim=-1).clamp_min(0.0)
    clearance = obstacle_surface_distance_m - vehicle_radius_m
    distance_penalty = (falloff_distance_m - clearance).clamp_min(0.0).square()
    intersection_barrier = functional.softplus(barrier_slope * clearance)
    cost = closing_speed * distance_penalty + barrier_scale * intersection_barrier
    return cost * obstacle_known.to(cost.dtype)


def depthnav_yaw_alignment_reward(
    yaw_error_rad: torch.Tensor,
    target_distance_m: torch.Tensor,
    disable_within_m: float = 1.0,
) -> torch.Tensor:
    """Return DepthNav's cosine yaw reward, disabled near the goal."""

    if yaw_error_rad.shape != target_distance_m.shape:
        raise ValueError("yaw error and target distance must have matching shapes")
    if disable_within_m < 0.0:
        raise ValueError("disable_within_m must be non-negative")
    aligned = torch.cos(yaw_error_rad) - 1.0
    return torch.where(target_distance_m < disable_within_m, torch.zeros_like(aligned), aligned)


def depthnav_acceleration_cost(acceleration_w: torch.Tensor) -> torch.Tensor:
    """Return DepthNav's squared physical linear-acceleration norm."""

    return acceleration_w.square().sum(dim=-1)


def depthnav_jerk_cost(
    acceleration_w: torch.Tensor, previous_acceleration_w: torch.Tensor, dt: float
) -> torch.Tensor:
    """Return DepthNav's squared physical jerk norm."""

    if acceleration_w.shape != previous_acceleration_w.shape:
        raise ValueError("current and previous acceleration must have matching shapes")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    jerk_w = (acceleration_w - previous_acceleration_w) / dt
    return jerk_w.square().sum(dim=-1)


def altitude_margin_cost(
    altitude_m: torch.Tensor,
    minimum_altitude_m: float,
    influence_height_m: float,
) -> torch.Tensor:
    """Return a bounded cost before the aircraft reaches the strict flight floor."""

    if influence_height_m <= 0.0:
        raise ValueError("influence_height_m must be positive")
    proximity = (
        (minimum_altitude_m + influence_height_m - altitude_m) / influence_height_m
    ).clamp(0.0, 1.0)
    return proximity.square()
