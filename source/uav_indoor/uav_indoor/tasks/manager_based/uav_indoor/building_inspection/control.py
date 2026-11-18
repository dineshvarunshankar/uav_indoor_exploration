"""Pure control transforms shared by the Isaac action term and unit tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VelocityLimits:
    v_max_xy: float
    v_max_z: float
    yaw_rate_max: float
    accel_max_xy: float
    accel_max_z: float
    yaw_accel_max: float


def bounded_velocity_step(
    raw_action: torch.Tensor,
    previous: torch.Tensor,
    dt: float,
    limits: VelocityLimits,
) -> torch.Tensor:
    """Apply tanh bounds and translational/yaw slew limits."""

    squashed = torch.tanh(raw_action)
    desired = torch.empty_like(squashed)
    desired[:, 0] = squashed[:, 0] * limits.v_max_xy
    desired[:, 1] = squashed[:, 1] * limits.v_max_xy
    desired[:, 2] = squashed[:, 2] * limits.v_max_z
    desired[:, 3] = squashed[:, 3] * limits.yaw_rate_max

    result = previous.clone()
    linear_delta = desired[:, :3] - previous[:, :3]
    horizontal_norm = torch.linalg.vector_norm(linear_delta[:, :2], dim=-1).clamp_min(1.0e-6)
    horizontal_scale = torch.clamp(limits.accel_max_xy * dt / horizontal_norm, max=1.0)
    result[:, :2] += linear_delta[:, :2] * horizontal_scale[:, None]
    result[:, 2] += linear_delta[:, 2].clamp(-limits.accel_max_z * dt, limits.accel_max_z * dt)
    yaw_delta = desired[:, 3] - previous[:, 3]
    result[:, 3] += yaw_delta.clamp(-limits.yaw_accel_max * dt, limits.yaw_accel_max * dt)
    return result
