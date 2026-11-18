"""Isolated entry-skill episodes: one selected opening, one exterior start pose."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class EntryTrainingCfg:
    """Start distribution for the entry skill, widened by success."""

    enabled: bool = False

    truth_facade_normal_w: tuple[float, float, float] = (-1.0, 0.0, 0.0)

    lateral_offset_m: tuple[float, float] = (-2.5, 2.5)
    vertical_offset_m: tuple[float, float] = (-1.5, 1.5)
    yaw_error_rad: tuple[float, float] = (-0.6, 0.6)
    standoff_m: tuple[float, float] = (2.0, 6.0)

    initial_width: float = 0.5
    width_step: float = 0.1
    promote_success_rate: float = 0.7
    minimum_episodes: int = 32

    min_height_m: float = 2.5

    success_crossing_margin_m: float = 0.5

    hold_selected_opening: bool = True

    def __post_init__(self) -> None:
        for name in ("lateral_offset_m", "vertical_offset_m", "yaw_error_rad", "standoff_m"):
            low, high = getattr(self, name)
            assert low < high, f"{name} bounds are not ordered"
        assert self.standoff_m[0] > 0.0, "a start pose behind the facade is not an entry episode"
        assert 0.0 < self.initial_width <= 1.0, "the initial curriculum width must be a fraction"


@dataclass
class EntryStartCurriculum:
    """Widen the start distribution only after repeated success at the present width."""

    cfg: EntryTrainingCfg
    width: float = field(default=0.0)
    _successes: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.width == 0.0:
            self.width = self.cfg.initial_width

    def record(self, successes: list[bool]) -> float:
        """Record finished episodes and return the width to use for the next resets."""

        self._successes.extend(bool(value) for value in successes)
        if len(self._successes) >= self.cfg.minimum_episodes:
            rate = sum(self._successes) / len(self._successes)
            if rate >= self.cfg.promote_success_rate and self.width < 1.0:
                self.width = min(1.0, self.width + self.cfg.width_step)
            self._successes.clear()
        return self.width


def _scaled(bounds: tuple[float, float], width: float, samples: torch.Tensor) -> torch.Tensor:
    """Map samples in [0, 1) onto a fraction of the configured bounds, centred."""

    low, high = bounds
    centre = 0.5 * (low + high)
    half = 0.5 * (high - low) * width
    return centre + (2.0 * samples - 1.0) * half


def sample_entry_start(
    opening_position_w: torch.Tensor,
    facade_normal_w: torch.Tensor,
    cfg: EntryTrainingCfg,
    width: float,
    samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return start positions and yaws that face the selected opening."""

    assert samples.shape[-1] == 4, "four sample dimensions are required"
    tangent = torch.stack(
        (-facade_normal_w[:, 1], facade_normal_w[:, 0], torch.zeros_like(facade_normal_w[:, 0])), dim=-1
    )
    lateral = _scaled(cfg.lateral_offset_m, width, samples[:, 0])
    vertical = _scaled(cfg.vertical_offset_m, width, samples[:, 1])
    standoff = _scaled(cfg.standoff_m, width, samples[:, 2])
    yaw_error = _scaled(cfg.yaw_error_rad, width, samples[:, 3])

    position = (
        opening_position_w
        + tangent * lateral[:, None]
        + facade_normal_w * standoff[:, None]
    )
    position[:, 2] = (opening_position_w[:, 2] + vertical).clamp_min(cfg.min_height_m)

    facing = torch.atan2(-facade_normal_w[:, 1], -facade_normal_w[:, 0])
    return position, facing + yaw_error
