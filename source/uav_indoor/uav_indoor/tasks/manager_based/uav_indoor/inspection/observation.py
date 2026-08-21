"""Primary-person observation timing for one opening.

The mission observes one confirmed person for ten accumulated seconds. The timer counts
only valid time, so it pauses instead of resetting when the evidence weakens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

WITNESS_FEATURES = 6  # identifier, last-seen time, confidence, and box centre and size


@dataclass(frozen=True)
class ObservationCfg:
    """Timing and validity constants.

    PROVISIONAL. ``centre_tolerance`` is horizontal only, because the aircraft centres a
    person by yaw alone. Vertical centring would need forward or vertical motion, which
    the first version does not use.
    """

    hold_target_s: float = 10.0
    centre_tolerance: float = 0.15
    min_confidence: float = 0.35
    max_track_age_s: float = 0.5
    lost_timeout_s: float = 2.0
    max_witnesses: int = 8


class PrimaryPersonObservation:
    """Batched ten-second hold on one primary person, with a record of the others."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cfg: ObservationCfg | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or ObservationCfg()
        witnesses = self.cfg.max_witnesses

        self.primary_id = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.held_s = torch.zeros(num_envs, device=self.device)
        self.lost_s = torch.zeros(num_envs, device=self.device)
        self.complete = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.centred = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.primary_visible = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.primary_offset = torch.zeros(num_envs, device=self.device)

        self.witness_id = torch.full((num_envs, witnesses), -1, dtype=torch.long, device=self.device)
        self.witness_time_s = torch.zeros(num_envs, witnesses, device=self.device)
        self.witness_confidence = torch.zeros(num_envs, witnesses, device=self.device)
        self.witness_box = torch.zeros(num_envs, witnesses, 4, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.primary_id[env_ids] = -1
        self.held_s[env_ids] = 0.0
        self.lost_s[env_ids] = 0.0
        self.complete[env_ids] = False
        self.centred[env_ids] = False
        self.primary_visible[env_ids] = False
        self.primary_offset[env_ids] = 0.0
        self.witness_id[env_ids] = -1
        self.witness_time_s[env_ids] = 0.0
        self.witness_confidence[env_ids] = 0.0
        self.witness_box[env_ids] = 0.0

    def clear(self, mask: torch.Tensor) -> None:
        """Release the hold where ``mask`` is true, without a host synchronization."""

        keep = ~mask
        self.primary_id = torch.where(mask, torch.full_like(self.primary_id, -1), self.primary_id)
        self.held_s = self.held_s * keep
        self.lost_s = self.lost_s * keep
        self.complete = self.complete & keep
        self.centred = self.centred & keep
        self.primary_visible = self.primary_visible & keep
        self.primary_offset = self.primary_offset * keep
        self.witness_id = torch.where(mask[:, None], torch.full_like(self.witness_id, -1), self.witness_id)
        self.witness_time_s = self.witness_time_s * keep[:, None]
        self.witness_confidence = self.witness_confidence * keep[:, None]
        self.witness_box = self.witness_box * keep[:, None, None]

    def update(
        self,
        dt: float,
        track_id: torch.Tensor,
        box_xywhn: torch.Tensor,
        confidence: torch.Tensor,
        age_s: torch.Tensor,
        valid: torch.Tensor,
        avoidance_active: torch.Tensor,
        time_s: float,
        enabled: torch.Tensor | None = None,
    ) -> None:
        """Advance the hold from the person tracks of one step."""

        if enabled is None:
            enabled = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        confirmed = valid & (confidence >= self.cfg.min_confidence) & (age_s <= self.cfg.max_track_age_s)

        seen = ((track_id == self.primary_id[:, None]) & confirmed).any(dim=1)
        has_primary = self.primary_id >= 0
        self.lost_s = torch.where(seen, torch.zeros_like(self.lost_s), self.lost_s + dt * has_primary)

        # A permanently lost primary releases the hold. The next person restarts the timer.
        lost = has_primary & (self.lost_s > self.cfg.lost_timeout_s)
        offered = torch.where(confirmed, confidence, torch.full_like(confidence, -torch.inf))
        best = offered.max(dim=1)
        replace = enabled & ~self.complete & (~has_primary | lost) & torch.isfinite(best.values)
        self.primary_id = torch.where(replace, track_id.gather(1, best.indices[:, None]).squeeze(1), self.primary_id)
        self.held_s = torch.where(replace, torch.zeros_like(self.held_s), self.held_s)
        self.lost_s = torch.where(replace, torch.zeros_like(self.lost_s), self.lost_s)

        primary = (track_id == self.primary_id[:, None]) & confirmed & (self.primary_id >= 0)[:, None]
        self.primary_visible = primary.any(dim=1)
        slot = primary.long().argmax(dim=1)
        centre_x = box_xywhn.gather(1, slot[:, None, None].expand(-1, -1, 4)).squeeze(1)[:, 0]
        self.primary_offset = torch.where(
            self.primary_visible, centre_x - 0.5, torch.zeros_like(self.primary_offset)
        )
        self.centred = self.primary_visible & (self.primary_offset.abs() <= self.cfg.centre_tolerance)

        # Valid time only. Every other state pauses the clock and keeps the total.
        counting = enabled & self.centred & ~avoidance_active & ~self.complete
        self.held_s = self.held_s + dt * counting
        self.complete = self.complete | (self.held_s >= self.cfg.hold_target_s)

        self._record_witnesses(confirmed & ~primary, track_id, box_xywhn, confidence, time_s, enabled)

    def _record_witnesses(
        self,
        other: torch.Tensor,
        track_id: torch.Tensor,
        box_xywhn: torch.Tensor,
        confidence: torch.Tensor,
        time_s: float,
        enabled: torch.Tensor,
    ) -> None:
        """Keep a record of the other visible people. No second hold starts for them.

        The table holds the first ``max_witnesses`` distinct people. A later person is
        dropped rather than replacing a recorded one, because this is a record of who was
        present, not a ranking.
        """

        other = other & enabled[:, None]
        known = (self.witness_id[:, :, None] == track_id[:, None, :]) & other[:, None, :]
        matched_slot = known.any(dim=2)
        source = known.long().argmax(dim=2)
        self.witness_time_s = torch.where(
            matched_slot, torch.full_like(self.witness_time_s, time_s), self.witness_time_s
        )
        self.witness_confidence = torch.where(matched_slot, confidence.gather(1, source), self.witness_confidence)
        self.witness_box = torch.where(
            matched_slot[..., None], box_xywhn.gather(1, source[..., None].expand(-1, -1, 4)), self.witness_box
        )

        new = other & ~known.any(dim=1)
        free = self.witness_id < 0
        slot_rank = free.cumsum(dim=1) - 1
        track_rank = new.cumsum(dim=1) - 1
        pairing = free[:, :, None] & new[:, None, :] & (slot_rank[:, :, None] == track_rank[:, None, :])
        written = pairing.any(dim=2)
        source = pairing.long().argmax(dim=2)
        self.witness_id = torch.where(written, track_id.gather(1, source), self.witness_id)
        self.witness_time_s = torch.where(written, torch.full_like(self.witness_time_s, time_s), self.witness_time_s)
        self.witness_confidence = torch.where(written, confidence.gather(1, source), self.witness_confidence)
        self.witness_box = torch.where(
            written[..., None], box_xywhn.gather(1, source[..., None].expand(-1, -1, 4)), self.witness_box
        )

    @property
    def witness_count(self) -> torch.Tensor:
        return (self.witness_id >= 0).sum(dim=1)

    def observation(self) -> torch.Tensor:
        """Actor features: held fraction, centring, primary validity, and completion."""

        return torch.stack(
            (
                (self.held_s / self.cfg.hold_target_s).clamp(0.0, 1.0),
                self.centred.float(),
                self.primary_visible.float(),
                self.complete.float(),
            ),
            dim=-1,
        )
