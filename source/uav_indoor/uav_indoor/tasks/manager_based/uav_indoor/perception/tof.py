"""Sparse ToF sampling and entry-corridor evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TofCfg:
    """Sampling, range limits, and failure modelling. Stride 4 keeps 6.25% of the frame."""

    stride: int = 4
    min_range_m: float = 0.2
    max_range_m: float = 6.0
    corridor_fraction: float = 0.35
    min_valid_fraction: float = 0.25
    range_quantile: float = 0.05
    sun_dropout_probability: float = 0.15
    point_dropout_probability: float = 0.10


@dataclass
class TofPoints:
    """Sparse metric returns for a camera batch, on the sensor device."""

    range_m: torch.Tensor  # (N, P) clamped to the valid window, zero where invalid
    valid: torch.Tensor  # (N, P)
    height: int
    width: int

    @property
    def valid_fraction(self) -> torch.Tensor:
        return self.valid.float().mean(dim=1)

    def image(self) -> torch.Tensor:
        """Return the sparse points reshaped to their sampling grid."""

        return self.range_m.view(-1, self.height, self.width)


class TofSampler:
    """Turn a rendered depth frame into the sparse, clamped returns of the real sensor."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cfg: TofCfg | None = None,
        seed: int = 0,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or TofCfg()
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)

    def sample(self, depth: torch.Tensor, randomize: bool = True) -> TofPoints:
        """Subsample, clamp, and drop returns. ``depth`` is ``(N, H, W)`` or ``(N, H, W, 1)`` in metres."""

        if depth.dim() == 4:
            depth = depth.squeeze(-1)
        stride = self.cfg.stride
        sparse = depth[:, stride // 2 :: stride, stride // 2 :: stride]
        height, width = sparse.shape[-2:]
        flat = sparse.reshape(depth.shape[0], -1)

        valid = torch.isfinite(flat) & (flat >= self.cfg.min_range_m) & (flat <= self.cfg.max_range_m)
        if randomize:
            kept = (
                torch.rand(flat.shape, device=self.device, generator=self._generator)
                >= self.cfg.point_dropout_probability
            )
            lit = (
                torch.rand(flat.shape[0], device=self.device, generator=self._generator)
                >= self.cfg.sun_dropout_probability
            )
            valid = valid & kept & lit[:, None]
        return TofPoints(torch.where(valid, flat, torch.zeros_like(flat)), valid, height, width)

    def corridor_evidence(self, points: TofPoints) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``tof_valid`` and ``tof_range_m``.

        A low quantile of the central window, so one noisy point decides nothing.
        Too few valid returns give a zero range, which can never pass the clearance test.
        """

        grid = points.valid.view(-1, points.height, points.width)
        ranges = points.image()
        margin_h = int(points.height * (1.0 - self.cfg.corridor_fraction) / 2)
        margin_w = int(points.width * (1.0 - self.cfg.corridor_fraction) / 2)
        centre_valid = grid[:, margin_h : points.height - margin_h, margin_w : points.width - margin_w]
        centre_range = ranges[:, margin_h : points.height - margin_h, margin_w : points.width - margin_w]

        centre_valid = centre_valid.flatten(1)
        centre_range = centre_range.flatten(1)
        enough = centre_valid.float().mean(dim=1) >= self.cfg.min_valid_fraction
        filled = torch.where(centre_valid, centre_range, torch.full_like(centre_range, self.cfg.max_range_m))
        near = torch.quantile(filled, self.cfg.range_quantile, dim=1)
        return enough, torch.where(enough, near, torch.zeros_like(near))
