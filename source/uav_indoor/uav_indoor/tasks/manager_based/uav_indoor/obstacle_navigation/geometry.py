"""Distances to the physical box and vertical-cylinder obstacle solids."""

from __future__ import annotations

import torch


def nearest_geometry(
    position_m: torch.Tensor,
    centres_m: torch.Tensor,
    sizes_m: torch.Tensor,
    active: torch.Tensor,
    cylinder_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return distance and offset to the nearest active physical solid."""

    if position_m.ndim != 2 or position_m.shape[-1] != 3:
        raise ValueError("position_m must have shape (N, 3)")
    if centres_m.shape != sizes_m.shape or centres_m.shape[:1] != position_m.shape[:1]:
        raise ValueError("centres_m and sizes_m must have shape (N, M, 3)")
    if active.shape != centres_m.shape[:2] or cylinder_slots.shape != (centres_m.shape[1],):
        raise ValueError("active or cylinder slot shape does not match geometry")

    point = position_m[:, None, :]
    half = sizes_m * 0.5
    box_closest = torch.maximum(torch.minimum(point, centres_m + half), centres_m - half)
    box_offset = box_closest - point

    relative = point - centres_m
    radial = torch.linalg.vector_norm(relative[..., :2], dim=-1, keepdim=True)
    radius = half[..., :1]
    radial_scale = torch.minimum(torch.ones_like(radial), radius / radial.clamp_min(1.0e-8))
    cylinder_xy = centres_m[..., :2] + relative[..., :2] * radial_scale
    cylinder_z = torch.maximum(
        torch.minimum(point[..., 2:3], centres_m[..., 2:3] + half[..., 2:3]),
        centres_m[..., 2:3] - half[..., 2:3],
    )
    cylinder_offset = torch.cat((cylinder_xy, cylinder_z), dim=-1) - point
    offsets = torch.where(cylinder_slots[None, :, None], cylinder_offset, box_offset)
    distances = torch.linalg.vector_norm(offsets, dim=-1)
    distances = torch.where(active, distances, torch.full_like(distances, torch.inf))
    nearest_distance, index = distances.min(dim=1)
    row = torch.arange(position_m.shape[0], device=position_m.device)
    nearest_offset = offsets[row, index]
    known = active.any(dim=1)
    nearest_distance = torch.where(known, nearest_distance, torch.zeros_like(nearest_distance))
    nearest_offset = torch.where(known[:, None], nearest_offset, torch.zeros_like(nearest_offset))
    return nearest_distance, nearest_offset, known
