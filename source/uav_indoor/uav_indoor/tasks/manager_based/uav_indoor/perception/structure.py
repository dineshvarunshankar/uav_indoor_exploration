"""Metric structure from bearings: facade plane fitting and incremental triangulation."""

from __future__ import annotations

import torch


def accumulate_rays(
    accumulator_a: torch.Tensor,
    accumulator_b: torch.Tensor,
    origin_w: torch.Tensor,
    ray_w: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add one bearing per slot to the running least-squares state."""

    projector = torch.eye(3, device=ray_w.device).expand(*ray_w.shape[:-1], 3, 3)
    projector = projector - ray_w.unsqueeze(-1) * ray_w.unsqueeze(-2)
    if weight is not None:
        projector = projector * weight[..., None, None]
    return (
        accumulator_a + projector,
        accumulator_b + (projector @ origin_w.unsqueeze(-1)).squeeze(-1),
    )


def solve_accumulated(
    accumulator_a: torch.Tensor, accumulator_b: torch.Tensor, min_conditioning: float = 0.008
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return triangulated positions and validity from the accumulated normal equations."""

    # Conditioning from singular values.
    spectrum = torch.linalg.svdvals(accumulator_a)
    conditioned = (spectrum[..., -1] / spectrum[..., 0].clamp_min(1e-9)) > min_conditioning

    regularised = accumulator_a + torch.eye(3, device=accumulator_a.device) * 1e-6
    solution = torch.linalg.solve(regularised, accumulator_b.unsqueeze(-1)).squeeze(-1)
    return solution, conditioned


def fit_vertical_plane(
    points_w: torch.Tensor, valid: torch.Tensor, prior_normal_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a vertical plane ``normal . x = offset`` through points on a facade."""

    weights = valid.float()
    count = weights.sum(dim=-1)
    total = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    centroid = (points_w * weights[..., None]).sum(dim=-2) / total

    centred = (points_w - centroid[..., None, :]) * weights[..., None]
    xx = (centred[..., 0] * centred[..., 0]).sum(dim=-1)
    yy = (centred[..., 1] * centred[..., 1]).sum(dim=-1)
    xy = (centred[..., 0] * centred[..., 1]).sum(dim=-1)

    # Smallest eigenvector of the 2x2 scatter matrix is the in-plane normal.
    difference = xx - yy
    angle = 0.5 * torch.atan2(2.0 * xy, difference)
    normal = torch.stack((torch.cos(angle + torch.pi / 2), torch.sin(angle + torch.pi / 2),
                          torch.zeros_like(angle)), dim=-1)

    # Orient the normal toward the approach side.
    flip = (normal * prior_normal_w).sum(dim=-1, keepdim=True) < 0.0
    normal = torch.where(flip, -normal, normal)

    usable = count >= 2
    normal = torch.where(usable[..., None], normal, prior_normal_w)
    offset = (normal * centroid).sum(dim=-1)
    return normal, offset, usable
