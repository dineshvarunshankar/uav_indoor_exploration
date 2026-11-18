"""Relative-disparity history and the reactive avoidance flag."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .midas import MIDAS_MODEL_PATH, MidasOnnxRunner, normalize_disparity


# Mutable Hydra configuration.
@dataclass
class DisparityCfg:
    """Rate, buffer shape, and the reactive threshold. Disparity is relative, never metric."""

    source: str = "midas"
    model_path: str = str(MIDAS_MODEL_PATH)
    period_s: float = 0.2
    history: int = 2
    height: int = 64
    width: int = 64
    centre_fraction: float = 0.4
    avoidance_threshold: float = 0.75


class DisparityMonitor:
    """Runs MiDaS at its deployed rate and keeps a short history for the actor."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cfg: DisparityCfg | None = None,
        runner: MidasOnnxRunner | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or DisparityCfg()
        if self.cfg.source not in ("midas", "rendered_inverse_depth"):
            raise ValueError(f"Unknown image source: {self.cfg.source}")
        self.runner = runner
        self.history = torch.zeros(num_envs, self.cfg.history, self.cfg.height, self.cfg.width, device=self.device)
        self.age_s = torch.full((num_envs,), torch.inf, device=self.device)
        self.avoidance_active = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._elapsed_s = self.cfg.period_s

    def load(self) -> MidasOnnxRunner:
        if self.runner is None:
            self.runner = MidasOnnxRunner(self.cfg.model_path, self.num_envs, self.device)
        return self.runner

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.history[env_ids] = 0.0
        self.age_s[env_ids] = torch.inf
        self.avoidance_active[env_ids] = False

    def tick(self, dt: float, rgb: torch.Tensor, depth: torch.Tensor | None = None) -> bool:
        """Advance the age and run the model on its own period."""

        self.age_s = self.age_s + dt
        self._elapsed_s += dt
        if self._elapsed_s + 1.0e-9 < self.cfg.period_s:
            return False
        self._elapsed_s %= self.cfg.period_s
        if self.cfg.source == "midas":
            raw = self.load().infer_batch(rgb[..., :3])
        else:
            if depth is None:
                raise ValueError("Rendered inverse depth requires camera depth")
            if depth.ndim == 4:
                depth = depth[..., 0]
            valid = torch.isfinite(depth) & (depth > 1.0e-6)
            inverse = torch.where(valid, depth.clamp_min(1.0e-6).reciprocal(), 0.0)
            raw = F.interpolate(inverse[:, None], size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
        self.update(normalize_disparity(raw))
        return True

    def update(self, disparity: torch.Tensor) -> None:
        """Push one normalized frame into the history and refresh the avoidance flag."""

        frame = F.adaptive_max_pool2d(
            disparity.unsqueeze(1), output_size=(self.cfg.height, self.cfg.width)
        ).squeeze(1)
        rolled = torch.roll(self.history, shifts=-1, dims=1)
        rolled[:, -1] = frame
        self.history = rolled
        self.age_s.zero_()
        self.avoidance_active = self.centre_disparity() >= self.cfg.avoidance_threshold

    def centre_disparity(self) -> torch.Tensor:
        """Highest disparity in the central window: near obstacles read high."""

        margin_h = int(self.cfg.height * (1.0 - self.cfg.centre_fraction) / 2)
        margin_w = int(self.cfg.width * (1.0 - self.cfg.centre_fraction) / 2)
        centre = self.history[:, -1, margin_h : self.cfg.height - margin_h, margin_w : self.cfg.width - margin_w]
        return centre.flatten(1).quantile(0.95, dim=1)

    def observation(self) -> torch.Tensor:
        """Return the disparity history as ``(N, T, H, W)`` for the actor encoder."""

        return self.history

    def freshness(self) -> torch.Tensor:
        """Validity flag and normalized age."""

        return torch.stack(
            ((self.age_s <= 2.0 * self.cfg.period_s).float(), (self.age_s / self.cfg.period_s).clamp(0.0, 10.0)),
            dim=-1,
        )


def ablate_disparity(
    image: torch.Tensor,
    mode: str,
    permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one controlled actor-input ablation without changing image shape."""

    if mode == "normal":
        return image
    if mode == "blank":
        return torch.zeros_like(image)
    if mode != "shuffled":
        raise ValueError(f"unknown disparity ablation: {mode}")
    pixels = image.shape[-2] * image.shape[-1]
    if permutation is None or permutation.numel() != pixels:
        raise ValueError("shuffled disparity requires one index for every pixel")
    flat = image.flatten(start_dim=-2)
    return flat.index_select(-1, permutation.to(image.device)).reshape_as(image)


def inverse_depth_alignment(
    disparity: torch.Tensor,
    metric_depth_m: torch.Tensor,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure per-image correlation with inverse metric depth for diagnostics."""

    if disparity.shape != metric_depth_m.shape or disparity.ndim != 4:
        raise ValueError("disparity and metric depth must have matching (N, C, H, W) shapes")
    finite = torch.isfinite(metric_depth_m) & (metric_depth_m > eps)
    inverse_depth = torch.where(finite, metric_depth_m.reciprocal(), torch.zeros_like(metric_depth_m))
    weights = finite.flatten(1).to(disparity.dtype)
    count = weights.sum(dim=1)
    safe_count = count.clamp_min(1.0)
    disparity_flat = disparity.flatten(1)
    inverse_flat = inverse_depth.flatten(1)
    disparity_mean = (disparity_flat * weights).sum(dim=1) / safe_count
    inverse_mean = (inverse_flat * weights).sum(dim=1) / safe_count
    disparity_delta = (disparity_flat - disparity_mean[:, None]) * weights
    inverse_delta = (inverse_flat - inverse_mean[:, None]) * weights
    numerator = (disparity_delta * inverse_delta).sum(dim=1)
    denominator = torch.sqrt(
        disparity_delta.square().sum(dim=1) * inverse_delta.square().sum(dim=1)
    )
    valid = (count >= 2) & (denominator > eps)
    correlation = torch.where(valid, numerator / denominator.clamp_min(eps), torch.zeros_like(numerator))
    return correlation, valid
