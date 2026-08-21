"""Relative-disparity history and the reactive avoidance flag."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .midas import MIDAS_MODEL_PATH, MidasOnnxRunner, normalize_disparity


@dataclass(frozen=True)
class DisparityCfg:
    """Rate, buffer shape, and the reactive threshold. Disparity is relative, never metric."""

    model_path: str = str(MIDAS_MODEL_PATH)
    period_s: float = 0.2
    history: int = 2
    height: int = 64
    width: int = 64
    centre_fraction: float = 0.4
    avoidance_threshold: float = 0.75
    frame_drop_probability: float = 0.03
    photometric_randomization: bool = True


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

    def tick(self, dt: float, rgb: torch.Tensor) -> bool:
        """Advance the age and run the model on its own period."""

        self.age_s = self.age_s + dt
        self._elapsed_s += dt
        if self._elapsed_s + 1.0e-9 < self.cfg.period_s:
            return False
        self._elapsed_s %= self.cfg.period_s
        frames = self._randomize(rgb[..., :3]) if self.cfg.photometric_randomization else rgb[..., :3]
        self.update(normalize_disparity(self.load().infer_batch(frames)))
        return True

    def update(self, disparity: torch.Tensor) -> None:
        """Push one normalized frame into the history and refresh the avoidance flag."""

        frame = F.interpolate(
            disparity.unsqueeze(1), size=(self.cfg.height, self.cfg.width), mode="bilinear", align_corners=False
        ).squeeze(1)
        kept = torch.rand(self.num_envs, device=self.device) >= self.cfg.frame_drop_probability
        rolled = torch.roll(self.history, shifts=-1, dims=1)
        rolled[:, -1] = frame
        self.history = torch.where(kept[:, None, None, None], rolled, self.history)
        self.age_s = torch.where(kept, torch.zeros_like(self.age_s), self.age_s)
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

    def _randomize(self, rgb: torch.Tensor) -> torch.Tensor:
        image = rgb.float() / 255.0
        gain = torch.empty(self.num_envs, 1, 1, 1, device=self.device).uniform_(0.65, 1.35)
        gamma = torch.empty(self.num_envs, 1, 1, 1, device=self.device).uniform_(0.80, 1.20)
        colour = torch.empty(self.num_envs, 1, 1, 3, device=self.device).uniform_(0.90, 1.10)
        noise = torch.randn_like(image) * 0.01
        image = (image * gain * colour).clamp(0.0, 1.0).pow(gamma).add(noise).clamp(0.0, 1.0)
        return (image * 255.0).to(torch.uint8)
