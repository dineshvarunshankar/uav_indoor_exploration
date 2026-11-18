"""Small, deployment-reproducible feature transforms for the local-flight actor."""

from __future__ import annotations

import torch
import torch.nn.functional as F


ACTOR_VECTOR_SCHEMA_VERSION = 2
ACTOR_VECTOR_WIDTH = 41
_NORMALIZED_TOLERANCE = 1.0e-6


def _normalized(name: str, values: torch.Tensor) -> torch.Tensor:
    """Enforce the raw actor-observation contract before model standardization."""

    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    if torch.any(values < -1.0 - _NORMALIZED_TOLERANCE) or torch.any(
        values > 1.0 + _NORMALIZED_TOLERANCE
    ):
        raise ValueError(f"{name} must be in [-1, 1]")
    return values


def state_features(
    linear_velocity_b: torch.Tensor,
    angular_velocity_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    height_above_ground_m: torch.Tensor,
    state_fresh: torch.Tensor,
    linear_scale_m_s: float = 3.0,
    angular_scale_rad_s: float = 3.0,
    height_scale_m: float = 10.0,
) -> torch.Tensor:
    """Return bounded vehicle state that the onboard estimator supplies."""

    if linear_scale_m_s <= 0.0 or angular_scale_rad_s <= 0.0 or height_scale_m <= 0.0:
        raise ValueError("state feature scales must be positive")
    linear = (linear_velocity_b / linear_scale_m_s).clamp(-1.0, 1.0)
    angular = (angular_velocity_b / angular_scale_rad_s).clamp(-1.0, 1.0)
    gravity = F.normalize(projected_gravity_b, dim=-1, eps=1.0e-6)
    height = (height_above_ground_m / height_scale_m).clamp(-1.0, 1.0)
    return _normalized(
        "state features",
        torch.cat(
            (linear, angular, gravity, height[:, None], state_fresh.float()[:, None]),
            dim=-1,
        ),
    )


def reference_features(
    offset_b: torch.Tensor,
    yaw_error: torch.Tensor,
    distance_scale_m: float = 50.0,
) -> torch.Tensor:
    """Return body-frame waypoint direction, normalized distance, and yaw error."""

    if distance_scale_m <= 0.0:
        raise ValueError("reference distance scale must be positive")
    distance = torch.linalg.vector_norm(offset_b, dim=-1, keepdim=True)
    direction = offset_b / distance.clamp_min(1.0e-6)
    normalized_distance = (distance / distance_scale_m).clamp(0.0, 1.0)
    yaw = torch.stack((torch.cos(yaw_error), torch.sin(yaw_error)), dim=-1)
    return _normalized(
        "reference features",
        torch.cat((direction, normalized_distance, yaw), dim=-1),
    )


def motion_mode_features(phase: torch.Tensor) -> torch.Tensor:
    """Collapse mission bookkeeping phases into five local-control modes."""

    phase_to_mode = torch.tensor(
        (0, 1, 2, 2, 2, 3, 4, 1, 1, 1, 1),
        dtype=torch.long,
        device=phase.device,
    )
    mode = phase_to_mode[phase.long().clamp(0, len(phase_to_mode) - 1)]
    return _normalized(
        "motion-mode features",
        F.one_hot(mode, num_classes=5).to(dtype=torch.float32),
    )


def selected_detection_features(
    box_xywhn: torch.Tensor,
    confidence: torch.Tensor,
    age_s: torch.Tensor,
    valid: torch.Tensor,
    target_changed: torch.Tensor,
    age_scale_s: float = 1.0,
) -> torch.Tensor:
    """Return the committed opening box with freshness and identity-change state."""

    if age_scale_s <= 0.0:
        raise ValueError("detection age scale must be positive")
    measured = torch.cat(
        (
            box_xywhn.clamp(0.0, 1.0),
            confidence.clamp(0.0, 1.0)[:, None],
            (age_s / age_scale_s).clamp(0.0, 1.0)[:, None],
            valid.float()[:, None],
        ),
        dim=-1,
    )
    measured = torch.where(valid[:, None], measured, torch.zeros_like(measured))
    return _normalized(
        "opening-detection features",
        torch.cat((measured, target_changed.float()[:, None]), dim=-1),
    )


def disparity_image(history: torch.Tensor) -> torch.Tensor:
    """Return the latest square disparity frame for the learned encoder."""

    if history.ndim != 4 or history.shape[1] < 1:
        raise ValueError(f"disparity history must have shape (N, T, H, W), got {tuple(history.shape)}")
    latest = history[:, -1:]
    if not torch.isfinite(latest).all():
        raise ValueError("disparity image must be finite")
    return latest.clamp(0.0, 1.0)


def disparity_freshness_features(freshness: torch.Tensor) -> torch.Tensor:
    """Return disparity validity and bounded age without image statistics."""

    if freshness.ndim != 2 or freshness.shape[1] != 2:
        raise ValueError(f"disparity freshness must have shape (N, 2), got {tuple(freshness.shape)}")
    return _normalized(
        "disparity-freshness features",
        torch.stack(
            (freshness[:, 0].clamp(0.0, 1.0), (freshness[:, 1] / 10.0).clamp(0.0, 1.0)),
            dim=-1,
        ),
    )

def tof_features(
    range_m: torch.Tensor,
    valid: torch.Tensor,
    clear: torch.Tensor,
    point_count: torch.Tensor,
    threshold_configured: torch.Tensor,
    max_range_m: float,
    required_points: int,
) -> torch.Tensor:
    """Return selected-opening ToF evidence and explicit decision state."""

    if max_range_m <= 0.0 or required_points <= 0:
        raise ValueError("ToF range and point scales must be positive")
    return _normalized(
        "ToF features",
        torch.stack(
        (
            (range_m / max_range_m).clamp(0.0, 1.0),
            valid.float(),
            clear.float(),
            (point_count.float() / required_points).clamp(0.0, 1.0),
            threshold_configured.float(),
        ),
        dim=-1,
        ),
    )


def previous_action_features(action: torch.Tensor) -> torch.Tensor:
    """Return the last policy action in the nominal action range."""

    return _normalized("previous-action features", action.clamp(-1.0, 1.0))


def concatenate_policy_features(*blocks: torch.Tensor) -> torch.Tensor:
    """Concatenate the fixed actor blocks without adding hidden bookkeeping."""

    features = torch.cat(blocks, dim=-1)
    if features.shape[-1] != ACTOR_VECTOR_WIDTH:
        raise ValueError(
            f"actor vector schema {ACTOR_VECTOR_SCHEMA_VERSION} requires "
            f"{ACTOR_VECTOR_WIDTH} values, got {features.shape[-1]}"
        )
    return _normalized("actor vector", features)
