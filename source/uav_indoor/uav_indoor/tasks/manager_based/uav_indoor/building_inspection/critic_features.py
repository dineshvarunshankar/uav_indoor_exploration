"""Fixed semantic privileged features shared by all training stages."""

from __future__ import annotations

import torch
import torch.nn.functional as F


CRITIC_PRIVILEGED_SCHEMA_VERSION = 1
CRITIC_PRIVILEGED_WIDTH = 9


def privileged_navigation_features(
    target_offset_w: torch.Tensor,
    clearance_direction_w: torch.Tensor,
    clearance_m: torch.Tensor,
    clearance_known: torch.Tensor,
    target_scale_m: float = 50.0,
    clearance_scale_m: float = 10.0,
) -> torch.Tensor:
    """Encode target geometry and one known clearance direction for the critic."""

    if target_offset_w.ndim != 2 or target_offset_w.shape[-1] != 3:
        raise ValueError(f"target_offset_w must have shape (N, 3), got {tuple(target_offset_w.shape)}")
    if clearance_direction_w.shape != target_offset_w.shape:
        raise ValueError(
            "clearance_direction_w must have the same shape as target_offset_w, "
            f"got {tuple(clearance_direction_w.shape)}"
        )
    batch = target_offset_w.shape[0]
    if clearance_m.shape != (batch,) or clearance_known.shape != (batch,):
        raise ValueError("clearance_m and clearance_known must have shape (N,)")
    if target_scale_m <= 0.0 or clearance_scale_m <= 0.0:
        raise ValueError("feature scales must be positive")
    if not torch.isfinite(target_offset_w).all() or not torch.isfinite(clearance_direction_w).all():
        raise ValueError("critic vector inputs must be finite")
    if not torch.isfinite(clearance_m).all():
        raise ValueError("critic clearance must be finite")

    target_distance = torch.linalg.vector_norm(target_offset_w, dim=-1)
    target_direction = F.normalize(target_offset_w, dim=-1, eps=1.0e-6)
    clearance_direction = F.normalize(clearance_direction_w, dim=-1, eps=1.0e-6)
    known = clearance_known.bool()
    clearance_direction = torch.where(
        known[:, None], clearance_direction, torch.zeros_like(clearance_direction)
    )
    clearance = torch.where(
        known,
        (clearance_m / clearance_scale_m).clamp(0.0, 1.0),
        torch.zeros_like(clearance_m),
    )
    features = torch.cat(
        (
            target_direction,
            (target_distance / target_scale_m).clamp(0.0, 1.0)[:, None],
            clearance_direction,
            clearance[:, None],
            known.float()[:, None],
        ),
        dim=-1,
    )
    if features.shape[-1] != CRITIC_PRIVILEGED_WIDTH:
        raise RuntimeError(f"critic privileged schema requires {CRITIC_PRIVILEGED_WIDTH} values")
    return features
