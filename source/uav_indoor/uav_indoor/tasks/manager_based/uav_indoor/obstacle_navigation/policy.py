"""Fixed actor-observation adapter for obstacle-navigation pretraining."""

from __future__ import annotations

import torch

_CLIP_SENSITIVE_INDICES = (*range(6), 9, 14, 31, *range(37, 41))


_WIDTHS = {
    "state": 11,
    "reference": 6,
    "motion_mode": 5,
    "disparity_freshness": 2,
    "previous_action": 4,
}


def navigation_policy_features(
    state: torch.Tensor,
    reference: torch.Tensor,
    motion_mode: torch.Tensor,
    disparity_freshness: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    """Return the shared 41-value vector input; disparity remains a 2D observation."""

    blocks = {
        "state": state,
        "reference": reference,
        "motion_mode": motion_mode,
        "disparity_freshness": disparity_freshness,
        "previous_action": previous_action,
    }
    batch = state.shape[0]
    for name, block in blocks.items():
        if block.ndim != 2 or block.shape[0] != batch or block.shape[1] != _WIDTHS[name]:
            raise ValueError(
                f"{name} must have shape (N, {_WIDTHS[name]}), got {tuple(block.shape)}"
            )
    opening_unavailable = state.new_zeros((batch, 8))
    tof_unavailable = state.new_zeros((batch, 5))
    features = torch.cat(
        (
            state,
            reference,
            motion_mode,
            opening_unavailable,
            disparity_freshness,
            tof_unavailable,
            previous_action,
        ),
        dim=-1,
    )
    if features.shape[-1] != 41:
        raise RuntimeError(f"navigation adapter produced {features.shape[-1]} values; expected 41")
    if not torch.isfinite(features).all():
        raise ValueError("policy features must be finite")
    if torch.any(features < -1.0 - 1.0e-6) or torch.any(features > 1.0 + 1.0e-6):
        raise ValueError("policy features must be in [-1, 1]")
    return features



def clipped_continuous_fraction(
    features: torch.Tensor,
    threshold: float = 1.0 - 1.0e-6,
) -> torch.Tensor:
    """Return the fraction of scale-limited continuous values at either bound."""

    if features.ndim != 2 or features.shape[1] != 41:
        raise ValueError(f"actor vector must have shape (N, 41), got {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise ValueError("actor vector must be finite")
    if threshold <= 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in (0, 1]")
    limited = features[:, _CLIP_SENSITIVE_INDICES]
    return (limited.abs() >= threshold).float().mean()
