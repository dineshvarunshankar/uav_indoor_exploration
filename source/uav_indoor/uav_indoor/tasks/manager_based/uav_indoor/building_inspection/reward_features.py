"""Unweighted reward features for building inspection."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def collision_features(
    max_contact_force: torch.Tensor, threshold: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return collision occurrence and contact-force excess over ``threshold``."""
    excess = (max_contact_force - threshold).clamp_min(0.0)
    return (excess > 0.0).to(max_contact_force.dtype), excess


def altitude_violation(altitude_m: torch.Tensor, minimum_m: float = 2.5) -> torch.Tensor:
    """Return metres below the required altitude."""
    return (minimum_m - altitude_m).clamp_min(0.0)


def action_change(current_action: torch.Tensor, previous_action: torch.Tensor) -> torch.Tensor:
    """Return squared action change summed across action axes."""
    return (current_action - previous_action).square().sum(dim=-1)


def body_rate_cost(angular_velocity_b: torch.Tensor) -> torch.Tensor:
    """Return squared body angular rate summed across roll, pitch, and yaw."""
    return angular_velocity_b.square().sum(dim=-1)


def yaw_acceleration_cost(
    current_yaw_rate: torch.Tensor, previous_yaw_rate: torch.Tensor, dt: float
) -> torch.Tensor:
    """Return squared yaw angular acceleration."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return ((current_yaw_rate - previous_yaw_rate) / dt).square()


def approach_danger(
    velocity_w: torch.Tensor,
    obstacle_direction_w: torch.Tensor,
    distance_m: torch.Tensor,
    known: torch.Tensor,
    influence_distance_m: float,
) -> torch.Tensor:
    """Return closing speed scaled by proximity to known geometry."""
    if influence_distance_m <= 0.0:
        raise ValueError("influence_distance_m must be positive")
    direction = functional.normalize(obstacle_direction_w, dim=-1, eps=1.0e-8)
    closing_speed = (velocity_w * direction).sum(dim=-1).clamp_min(0.0)
    proximity = (influence_distance_m - distance_m).clamp_min(0.0) / influence_distance_m
    return closing_speed * proximity * known.to(closing_speed.dtype)


def potential_progress(
    previous_value: torch.Tensor,
    current_value: torch.Tensor,
    previous_phase: torch.Tensor,
    current_phase: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return decrease in a phase potential per second and the next stored state."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    progress = (previous_value - current_value) / dt
    progress = torch.where(previous_phase == current_phase, progress, torch.zeros_like(progress))
    return progress, current_value.clone(), current_phase.clone()


def box_center_error(box_xywhn: torch.Tensor) -> torch.Tensor:
    """Return squared image-centre error from a normalized live RGB box."""
    offset = (box_xywhn[..., :2] - 0.5) / 0.5
    return offset.square().sum(dim=-1)


def fixed_pose_errors(
    position_w: torch.Tensor,
    target_position_w: torch.Tensor,
    yaw_w: torch.Tensor,
    target_yaw_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return position error in metres and wrapped yaw error in radians."""
    position_error = torch.linalg.vector_norm(target_position_w - position_w, dim=-1)
    yaw_error = torch.remainder(target_yaw_w - yaw_w + torch.pi, 2.0 * torch.pi) - torch.pi
    return position_error, yaw_error.abs()


def visibility_signal(valid: torch.Tensor) -> torch.Tensor:
    """Return one for a valid selected live opening track and zero otherwise."""
    return valid.to(dtype=torch.float32)


def gate_final_floor_completion(
    floor_delta: torch.Tensor,
    mission_done: torch.Tensor,
    mission_success: torch.Tensor,
) -> torch.Tensor:
    """Pay ordinary floors and pay the final floor only after successful coverage."""

    return torch.where(mission_done & ~mission_success, torch.zeros_like(floor_delta), floor_delta)
