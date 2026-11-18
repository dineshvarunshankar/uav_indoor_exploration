"""Sparse ToF sampling and entry-corridor evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Mutable Hydra configuration.
@dataclass
class TofCfg:
    """Sparse sampling and the fixed close-range ToF limits."""

    stride: int = 4
    min_range_m: float = 0.2
    max_range_m: float = 2.5
    # Maximum reliable ToF range, in meters.
    reliable_range_m: float = 2.5
    corridor_fraction: float = 0.35
    min_valid_fraction: float = 0.25
    depth_quantile: float = 0.75
    clearance_quantile: float = 0.05
    min_region_points: int = 3
    target_inner_scale: float = 0.60
    # Measured clearance threshold; None leaves authorization UNKNOWN.
    required_clearance_m: float | None = None


@dataclass
class TofPoints:
    """Sparse metric returns for a camera batch, on the sensor device."""

    range_m: torch.Tensor  # (N, P) clamped to the valid window, zero where invalid
    valid: torch.Tensor  # (N, P)
    height: int
    width: int
    points_xyn: torch.Tensor  # (N, P, 2) sample centres in the ToF image

    @property
    def valid_fraction(self) -> torch.Tensor:
        return self.valid.float().mean(dim=1)

    def image(self) -> torch.Tensor:
        """Return the sparse points reshaped to their sampling grid."""

        return self.range_m.view(-1, self.height, self.width)


@dataclass
class TofTargetEvidence:
    """Evidence from valid returns inside the eroded selected RGB box."""

    valid: torch.Tensor
    opening_depth_m: torch.Tensor
    clear: torch.Tensor
    target_projected: torch.Tensor
    opening_measured: torch.Tensor
    point_count: torch.Tensor
    threshold_configured: torch.Tensor


class TofSampler:
    """Turn a rendered depth frame into the sparse, clamped returns of the real sensor."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cfg: TofCfg | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or TofCfg()

    def sample(self, depth: torch.Tensor) -> TofPoints:
        """Keep sparse returns only from objects inside the close-range entry window."""

        if depth.dim() == 4:
            depth = depth.squeeze(-1)
        stride = self.cfg.stride
        sparse = depth[:, stride // 2 :: stride, stride // 2 :: stride]
        height, width = sparse.shape[-2:]
        flat = sparse.reshape(depth.shape[0], -1)

        valid = (
            torch.isfinite(flat)
            & (flat >= self.cfg.min_range_m)
            & (flat <= min(self.cfg.max_range_m, self.cfg.reliable_range_m))
        )
        source_height, source_width = depth.shape[-2:]
        rows = (
            torch.arange(stride // 2, source_height, stride, device=self.device) + 0.5
        ) / source_height
        cols = (
            torch.arange(stride // 2, source_width, stride, device=self.device) + 0.5
        ) / source_width
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        points_xyn = torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)
        points_xyn = points_xyn.expand(depth.shape[0], -1, -1)
        return TofPoints(
            torch.where(valid, flat, torch.zeros_like(flat)),
            valid,
            height,
            width,
            points_xyn,
        )

    @staticmethod
    def _median(values: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = valid.sum(dim=-1)
        ordered = torch.where(valid, values, torch.full_like(values, torch.inf)).sort(dim=-1).values
        index = torch.div((count - 1).clamp_min(0), 2, rounding_mode="floor")
        median = ordered.gather(1, index[:, None]).squeeze(1)
        return torch.where(torch.isfinite(median), median, torch.zeros_like(median)), count

    def target_evidence(
        self,
        points: TofPoints,
        projected_xyn: torch.Tensor,
        projected_valid: torch.Tensor,
        box_xywhn: torch.Tensor,
        target_valid: torch.Tensor,
    ) -> TofTargetEvidence:
        """Evaluate only measured returns inside the eroded selected RGB box."""

        delta = (projected_xyn - box_xywhn[:, None, :2]).abs()
        half = 0.5 * box_xywhn[:, None, 2:4].clamp_min(1.0e-6)
        inside = (delta <= half * self.cfg.target_inner_scale).all(dim=-1)
        projected_inside = projected_valid & inside
        measured_inside = points.valid & projected_inside

        point_count = measured_inside.sum(dim=-1)
        opening_measured = point_count >= self.cfg.min_region_points
        target_projected = target_valid & (
            projected_inside.sum(dim=-1) >= self.cfg.min_region_points
        )

        filled = torch.where(
            measured_inside,
            points.range_m,
            torch.full_like(points.range_m, torch.inf),
        )
        nearest = filled.amin(dim=-1)
        nearest = torch.where(torch.isfinite(nearest), nearest, torch.zeros_like(nearest))

        configured = self.cfg.required_clearance_m is not None
        threshold_configured = torch.full_like(target_valid, configured)
        valid = target_projected & opening_measured & threshold_configured
        if configured:
            clear = valid & (nearest >= float(self.cfg.required_clearance_m))
        else:
            clear = torch.zeros_like(valid)

        return TofTargetEvidence(
            valid=valid,
            opening_depth_m=nearest,
            clear=clear,
            target_projected=target_projected,
            opening_measured=opening_measured,
            point_count=point_count,
            threshold_configured=threshold_configured,
        )

    def corridor_evidence(self, points: TofPoints) -> tuple[torch.Tensor, torch.Tensor]:
        """Return validity and robust deep-range evidence in the central window."""

        grid = points.valid.view(-1, points.height, points.width)
        ranges = points.image()
        margin_h = int(points.height * (1.0 - self.cfg.corridor_fraction) / 2)
        margin_w = int(points.width * (1.0 - self.cfg.corridor_fraction) / 2)
        centre_valid = grid[:, margin_h : points.height - margin_h, margin_w : points.width - margin_w]
        centre_range = ranges[:, margin_h : points.height - margin_h, margin_w : points.width - margin_w]

        centre_valid = centre_valid.flatten(1)
        centre_range = centre_range.flatten(1)
        enough = centre_valid.float().mean(dim=1) >= self.cfg.min_valid_fraction
        filled = torch.where(
            centre_valid,
            centre_range,
            torch.full_like(centre_range, self.cfg.min_range_m),
        )
        deep = torch.quantile(filled, self.cfg.depth_quantile, dim=1)
        return enough, torch.where(enough, deep, torch.zeros_like(deep))

    def region_evidence(
        self,
        points: TofPoints,
        box_xywhn: torch.Tensor,
        region_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deep-range evidence only inside the associated active-opening region."""

        rows = (
            torch.arange(points.height, device=self.device, dtype=box_xywhn.dtype) + 0.5
        ) / points.height
        cols = (
            torch.arange(points.width, device=self.device, dtype=box_xywhn.dtype) + 0.5
        ) / points.width
        grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
        centres = torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)

        half_size = 0.5 * box_xywhn[:, None, 2:4]
        in_region = (
            (centres - box_xywhn[:, None, :2]).abs() <= half_size
        ).all(dim=-1)
        sample_count = in_region.sum(dim=-1)
        return_valid = points.valid & in_region
        valid_count = return_valid.sum(dim=-1)
        valid_fraction = valid_count.float() / sample_count.clamp_min(1).float()
        enough = (
            region_valid
            & (sample_count >= self.cfg.min_region_points)
            & (valid_fraction >= self.cfg.min_valid_fraction)
        )

        sorted_range = torch.where(
            return_valid,
            points.range_m,
            torch.full_like(points.range_m, torch.inf),
        ).sort(dim=-1).values
        quantile_index = torch.ceil(
            self.cfg.depth_quantile * (valid_count - 1).clamp_min(0).float()
        ).long()
        deep = sorted_range.gather(1, quantile_index[:, None]).squeeze(1)
        deep = torch.where(torch.isfinite(deep), deep, torch.zeros_like(deep))
        return enough, torch.where(enough, deep, torch.zeros_like(deep))

    def corridor_clearance(self, points: TofPoints) -> torch.Tensor:
        """Return a near-range entry-corridor diagnostic; disparity owns avoidance."""

        grid = points.valid.view(-1, points.height, points.width)
        ranges = points.image()
        margin_h = int(points.height * (1.0 - self.cfg.corridor_fraction) / 2)
        margin_w = int(points.width * (1.0 - self.cfg.corridor_fraction) / 2)
        centre_valid = grid[
            :, margin_h : points.height - margin_h, margin_w : points.width - margin_w
        ].flatten(1)
        centre_range = ranges[
            :, margin_h : points.height - margin_h, margin_w : points.width - margin_w
        ].flatten(1)
        filled = torch.where(
            centre_valid,
            centre_range,
            torch.full_like(centre_range, self.cfg.max_range_m),
        )
        return torch.quantile(filled, self.cfg.clearance_quantile, dim=1)
