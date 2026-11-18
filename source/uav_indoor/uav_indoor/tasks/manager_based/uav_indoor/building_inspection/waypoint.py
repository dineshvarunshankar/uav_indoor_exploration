"""Pure local-waypoint transforms shared by simulation and deployment."""

from __future__ import annotations

import torch


def heading_yaw_w(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Return heading from a scalar-first world-frame quaternion."""

    w, x, y, z = quaternion_wxyz.unbind(dim=-1)
    sine = 2.0 * (w * z + x * y)
    cosine = 1.0 - 2.0 * (y.square() + z.square())
    return torch.atan2(sine, cosine)


def world_to_heading_frame(
    vector_w: torch.Tensor, quaternion_wxyz: torch.Tensor
) -> torch.Tensor:
    """Rotate a world vector into the yaw-only frame used by policy actions."""

    yaw_w = heading_yaw_w(quaternion_wxyz)
    cosine = torch.cos(yaw_w)
    sine = torch.sin(yaw_w)
    return torch.stack(
        (
            cosine * vector_w[:, 0] + sine * vector_w[:, 1],
            -sine * vector_w[:, 0] + cosine * vector_w[:, 1],
            vector_w[:, 2],
        ),
        dim=-1,
    )


def update_stale_hold(
    *,
    position_w: torch.Tensor,
    yaw_w: torch.Tensor,
    input_fresh: torch.Tensor,
    held_position_w: torch.Tensor,
    held_yaw_w: torch.Tensor,
    hold_active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture one finite pose when input changes from valid to stale."""

    pose_finite = torch.isfinite(position_w).all(dim=-1) & torch.isfinite(yaw_w)
    capture = (~input_fresh) & (~hold_active) & pose_finite
    next_position = torch.where(capture[:, None], position_w, held_position_w)
    next_yaw = torch.where(capture, yaw_w, held_yaw_w)
    return next_position, next_yaw, ~input_fresh


def body_relative_position_target(
    position_w: torch.Tensor,
    yaw_w: torch.Tensor,
    action: torch.Tensor,
    reach_m: torch.Tensor,
    yaw_reach_rad: float,
    translation_allowed: torch.Tensor,
    hold_position_w: torch.Tensor,
    input_fresh: torch.Tensor,
    stale_hold_position_w: torch.Tensor | None = None,
    stale_hold_yaw_w: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one policy action into an environment-local PX4 position target."""

    clipped = action.clamp(-1.0, 1.0)
    cosine = torch.cos(yaw_w)
    sine = torch.sin(yaw_w)
    offset_b = clipped[:, :3] * reach_m[None, :]
    offset_w = torch.stack(
        (
            cosine * offset_b[:, 0] - sine * offset_b[:, 1],
            sine * offset_b[:, 0] + cosine * offset_b[:, 1],
            offset_b[:, 2],
        ),
        dim=-1,
    )

    moving_target_w = position_w + offset_w
    target_w = torch.where(translation_allowed[:, None], moving_target_w, hold_position_w)
    target_yaw = yaw_w + clipped[:, 3] * yaw_reach_rad

    safe_action = clipped.clone()
    safe_action[:, :3] = torch.where(
        translation_allowed[:, None],
        safe_action[:, :3],
        torch.zeros_like(safe_action[:, :3]),
    )

    if stale_hold_position_w is None:
        stale_hold_position_w = position_w
    if stale_hold_yaw_w is None:
        stale_hold_yaw_w = yaw_w
    target_w = torch.where(input_fresh[:, None], target_w, stale_hold_position_w)
    target_yaw = torch.where(input_fresh, target_yaw, stale_hold_yaw_w)
    safe_action = torch.where(input_fresh[:, None], safe_action, torch.zeros_like(safe_action))

    wrapped_yaw = torch.atan2(torch.sin(target_yaw), torch.cos(target_yaw))
    return target_w, wrapped_yaw, safe_action
