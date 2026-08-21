"""Fixed-capacity opening inventory: the long-term building record.

Entries never expire. Short-term image memory belongs to the trackers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch


class OpeningStatus(IntEnum):
    """Life cycle of one opening. ``INSPECTED`` and ``UNSAFE`` are terminal."""

    EMPTY = 0
    CANDIDATE = 1
    ACTIVE = 2
    INSPECTED = 3
    UNSAFE = 4


@dataclass(frozen=True)
class LedgerCfg:
    """Capacity and matching constants. PROVISIONAL, see the plan for the measurement method."""

    max_openings: int = 10
    merge_radius_m: float = 1.5
    proposal_merge_radius_m: float = 2.5
    max_proposal_age_s: float = 30.0


class OpeningLedger:
    """Batched opening inventory, one entry set per environment.

    When full, the highest-confidence openings win. An active target and a terminal result
    are never replaced: losing either would repeat an inspection or break an approach.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cfg: LedgerCfg | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or LedgerCfg()
        slots = self.cfg.max_openings
        self.max_openings = slots

        self.position_w = torch.zeros(num_envs, slots, 3, device=self.device)
        # Direction from the observing scan station to the opening. Measured, never labelled.
        self.approach_yaw = torch.zeros(num_envs, slots, device=self.device)
        self.size_wh = torch.zeros(num_envs, slots, 2, device=self.device)
        self.floor = torch.zeros(num_envs, slots, dtype=torch.long, device=self.device)
        self.status = torch.zeros(num_envs, slots, dtype=torch.long, device=self.device)
        self.confidence = torch.zeros(num_envs, slots, device=self.device)
        self.weight = torch.zeros(num_envs, slots, device=self.device)
        self.hits = torch.zeros(num_envs, slots, dtype=torch.long, device=self.device)
        self.last_seen_s = torch.zeros(num_envs, slots, device=self.device)
        self.source = torch.zeros(num_envs, slots, dtype=torch.long, device=self.device)
        self.map_version = torch.zeros(num_envs, slots, dtype=torch.long, device=self.device)
        self.opening_id = torch.full((num_envs, slots), -1, dtype=torch.long, device=self.device)
        self._next_id = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    @property
    def occupied(self) -> torch.Tensor:
        return self.status != int(OpeningStatus.EMPTY)

    @property
    def terminal(self) -> torch.Tensor:
        return self.status >= int(OpeningStatus.INSPECTED)

    @property
    def pending(self) -> torch.Tensor:
        """Candidate or active. A floor is complete when no entry is pending."""

        return (self.status == int(OpeningStatus.CANDIDATE)) | (self.status == int(OpeningStatus.ACTIVE))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.position_w[env_ids] = 0.0
        self.approach_yaw[env_ids] = 0.0
        self.size_wh[env_ids] = 0.0
        self.floor[env_ids] = 0
        self.status[env_ids] = int(OpeningStatus.EMPTY)
        self.confidence[env_ids] = 0.0
        self.weight[env_ids] = 0.0
        self.hits[env_ids] = 0
        self.last_seen_s[env_ids] = 0.0
        self.source[env_ids] = 0
        self.map_version[env_ids] = 0
        self.opening_id[env_ids] = -1
        self._next_id[env_ids] = 0

    def observe(
        self,
        position_w: torch.Tensor,
        approach_yaw: torch.Tensor,
        size_wh: torch.Tensor,
        floor: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        time_s: float,
        source: int = 0,
        map_version: torch.Tensor | None = None,
        merge_radius_m: float | None = None,
    ) -> None:
        """Merge or insert a batch of metric openings seen from one scan station."""

        radius = self.cfg.merge_radius_m if merge_radius_m is None else merge_radius_m
        matched_slot, matched = self._match(position_w, floor, valid, radius)

        # A match on a terminal entry is ignored: no duplicate, and no revival.
        terminal_match = matched & self.terminal.gather(1, matched_slot)
        merged = matched & ~terminal_match
        self._merge(matched_slot, merged, position_w, approach_yaw, size_wh, confidence, time_s)

        unplaced = self._insert_free(
            valid & ~matched, position_w, approach_yaw, size_wh, floor, confidence, time_s, source, map_version
        )
        self._replace_weakest(
            unplaced, position_w, approach_yaw, size_wh, floor, confidence, time_s, source, map_version
        )

    def insert_proposals(
        self,
        position_w: torch.Tensor,
        approach_yaw: torch.Tensor,
        size_wh: torch.Tensor,
        floor: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        proposal_map_version: torch.Tensor,
        proposal_time_s: torch.Tensor,
        active_map_version: torch.Tensor,
        time_s: float,
    ) -> torch.Tensor:
        """Insert offboard proposals. Returns the accepted mask.

        A proposal stays a proposal until onboard sensors validate it, so it enters as a
        candidate exactly like an onboard observation.
        """

        fresh = (time_s - proposal_time_s) <= self.cfg.max_proposal_age_s
        same_map = proposal_map_version == active_map_version[:, None]
        accepted = valid & fresh & same_map
        self.observe(
            position_w,
            approach_yaw,
            size_wh,
            floor,
            confidence,
            accepted,
            time_s,
            source=1,
            map_version=proposal_map_version,
            merge_radius_m=self.cfg.proposal_merge_radius_m,
        )
        return accepted

    def _match(
        self,
        position_w: torch.Tensor,
        floor: torch.Tensor,
        valid: torch.Tensor,
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the nearest occupied slot on the same floor, and whether it is inside the radius."""

        distance = torch.cdist(self.position_w, position_w)  # (N, M, D)
        same_floor = self.floor[:, :, None] == floor[:, None, :]
        allowed = self.occupied[:, :, None] & same_floor & valid[:, None, :] & (distance <= radius)
        distance = torch.where(allowed, distance, torch.full_like(distance, torch.inf))
        best = distance.min(dim=1)
        return best.indices, torch.isfinite(best.values)

    def _merge(
        self,
        slot: torch.Tensor,
        merged: torch.Tensor,
        position_w: torch.Tensor,
        approach_yaw: torch.Tensor,
        size_wh: torch.Tensor,
        confidence: torch.Tensor,
        time_s: float,
    ) -> None:
        """Confidence-weighted mean position. Several observations may merge into one slot."""

        weight = torch.where(merged, confidence, torch.zeros_like(confidence))
        added_weight = torch.zeros_like(self.weight).scatter_add(1, slot, weight)
        added_position = torch.zeros_like(self.position_w).scatter_add(
            1, slot[..., None].expand(-1, -1, 3), position_w * weight[..., None]
        )
        added_yaw = torch.zeros_like(self.approach_yaw).scatter_add(1, slot, approach_yaw * weight)
        added_size = torch.zeros_like(self.size_wh).scatter_add(
            1, slot[..., None].expand(-1, -1, 2), size_wh * weight[..., None]
        )
        added_hits = torch.zeros_like(self.hits).scatter_add(1, slot, merged.long())
        best_confidence = torch.zeros_like(self.confidence).scatter_reduce(
            1, slot, weight, reduce="amax", include_self=False
        )

        touched = added_hits > 0
        total = (self.weight + added_weight).clamp_min(1.0e-6)
        self.position_w = torch.where(
            touched[..., None],
            (self.position_w * self.weight[..., None] + added_position) / total[..., None],
            self.position_w,
        )
        self.approach_yaw = torch.where(
            touched, (self.approach_yaw * self.weight + added_yaw) / total, self.approach_yaw
        )
        self.size_wh = torch.where(
            touched[..., None],
            (self.size_wh * self.weight[..., None] + added_size) / total[..., None],
            self.size_wh,
        )
        self.weight = torch.where(touched, self.weight + added_weight, self.weight)
        self.confidence = torch.where(touched, torch.maximum(self.confidence, best_confidence), self.confidence)
        self.hits = self.hits + added_hits
        self.last_seen_s = torch.where(touched, torch.full_like(self.last_seen_s, time_s), self.last_seen_s)

    def _write(
        self,
        written: torch.Tensor,
        source_index: torch.Tensor,
        position_w: torch.Tensor,
        approach_yaw: torch.Tensor,
        size_wh: torch.Tensor,
        floor: torch.Tensor,
        confidence: torch.Tensor,
        time_s: float,
        source: int,
        map_version: torch.Tensor | None,
    ) -> None:
        """Overwrite the slots in ``written`` with the observation named by ``source_index``."""

        gathered_position = position_w.gather(1, source_index[..., None].expand(-1, -1, 3))
        gathered_size = size_wh.gather(1, source_index[..., None].expand(-1, -1, 2))
        gathered_confidence = confidence.gather(1, source_index)
        self.position_w = torch.where(written[..., None], gathered_position, self.position_w)
        self.size_wh = torch.where(written[..., None], gathered_size, self.size_wh)
        self.approach_yaw = torch.where(written, approach_yaw.gather(1, source_index), self.approach_yaw)
        self.floor = torch.where(written, floor.gather(1, source_index), self.floor)
        self.confidence = torch.where(written, gathered_confidence, self.confidence)
        self.weight = torch.where(written, gathered_confidence, self.weight)
        self.hits = torch.where(written, torch.ones_like(self.hits), self.hits)
        self.last_seen_s = torch.where(written, torch.full_like(self.last_seen_s, time_s), self.last_seen_s)
        self.status = torch.where(written, torch.full_like(self.status, int(OpeningStatus.CANDIDATE)), self.status)
        self.source = torch.where(written, torch.full_like(self.source, source), self.source)
        version = (
            torch.zeros_like(self.map_version)
            if map_version is None
            else map_version.gather(1, source_index)
        )
        self.map_version = torch.where(written, version, self.map_version)
        self.opening_id = torch.where(
            written, self._next_id[:, None] + (written.cumsum(dim=1) - 1).clamp_min(0), self.opening_id
        )
        self._next_id = self._next_id + written.sum(dim=1)

    def _insert_free(
        self,
        unplaced: torch.Tensor,
        position_w: torch.Tensor,
        approach_yaw: torch.Tensor,
        size_wh: torch.Tensor,
        floor: torch.Tensor,
        confidence: torch.Tensor,
        time_s: float,
        source: int,
        map_version: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fill empty slots in order. Returns the observations that found no empty slot."""

        free = ~self.occupied
        slot_rank = free.cumsum(dim=1) - 1
        observation_rank = unplaced.cumsum(dim=1) - 1
        pairing = (
            free[:, :, None]
            & unplaced[:, None, :]
            & (slot_rank[:, :, None] == observation_rank[:, None, :])
        )
        written = pairing.any(dim=2)
        self._write(
            written, pairing.long().argmax(dim=2), position_w, approach_yaw, size_wh, floor,
            confidence, time_s, source, map_version,
        )
        return unplaced & ~pairing.any(dim=1)

    def _replace_weakest(
        self,
        unplaced: torch.Tensor,
        position_w: torch.Tensor,
        approach_yaw: torch.Tensor,
        size_wh: torch.Tensor,
        floor: torch.Tensor,
        confidence: torch.Tensor,
        time_s: float,
        source: int,
        map_version: torch.Tensor | None,
    ) -> None:
        """Keep the highest-confidence openings when the ledger is full.

        Each round moves the strongest remaining observation into the weakest candidate it
        beats. An active target and a terminal result are never replaced. The round count
        is bounded by the observation count, never by the environment count.
        """

        detections = unplaced.shape[1]
        for _ in range(min(self.max_openings, detections)):
            offered = torch.where(unplaced, confidence, torch.full_like(confidence, -torch.inf))
            best = offered.max(dim=1)
            candidate = self.status == int(OpeningStatus.CANDIDATE)
            held = torch.where(candidate, self.confidence, torch.full_like(self.confidence, torch.inf))
            weakest = held.min(dim=1)
            replace = best.values > weakest.values
            written = replace[:, None] & (
                torch.arange(self.max_openings, device=self.device)[None, :] == weakest.indices[:, None]
            )
            self._write(
                written, best.indices[:, None].expand(-1, self.max_openings), position_w,
                approach_yaw, size_wh, floor, confidence, time_s, source, map_version,
            )
            consumed = replace[:, None] & (
                torch.arange(detections, device=self.device)[None, :] == best.indices[:, None]
            )
            unplaced = unplaced & ~consumed

    def mark(self, slot: torch.Tensor, status: OpeningStatus, mask: torch.Tensor | None = None) -> None:
        """Write one status per environment at ``slot``."""

        selected = torch.arange(self.max_openings, device=self.device)[None, :] == slot[:, None]
        if mask is not None:
            selected = selected & mask[:, None]
        self.status = torch.where(selected, torch.full_like(self.status, int(status)), self.status)

    def select_candidate(
        self,
        building_center_w: torch.Tensor,
        station_yaw: torch.Tensor,
        survey_direction: torch.Tensor,
        floor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next candidate slot in the survey direction, and its validity.

        Openings are ordered by angle about the building centre. ``survey_direction`` is
        +1 for counterclockwise and -1 for clockwise.
        """

        offset = self.position_w - building_center_w[:, None, :]
        angle = torch.atan2(offset[..., 1], offset[..., 0])
        ahead = torch.remainder((angle - station_yaw[:, None]) * survey_direction[:, None], 2.0 * math.pi)
        eligible = (self.status == int(OpeningStatus.CANDIDATE)) & (self.floor == floor[:, None])
        ahead = torch.where(eligible, ahead, torch.full_like(ahead, torch.inf))
        best = ahead.min(dim=1)
        return best.indices, torch.isfinite(best.values)

    def floor_complete(self, floor: torch.Tensor) -> torch.Tensor:
        """True when that floor holds no candidate and no active opening."""

        return ~(self.pending & (self.floor == floor[:, None])).any(dim=1)

    def building_complete(self) -> torch.Tensor:
        """True when every recorded opening has a terminal result."""

        return ~self.pending.any(dim=1)

    def observation(self, robot_position_w: torch.Tensor) -> torch.Tensor:
        """Fixed-width ledger summary for the actor.

        Four status counts, the active-target validity and its body-frame offset, and the
        range to the nearest candidate. Counts are normalized by capacity.
        """

        counts = torch.stack(
            [
                (self.status == int(status)).sum(dim=1).float() / self.max_openings
                for status in (
                    OpeningStatus.CANDIDATE,
                    OpeningStatus.ACTIVE,
                    OpeningStatus.INSPECTED,
                    OpeningStatus.UNSAFE,
                )
            ],
            dim=-1,
        )
        active = self.status == int(OpeningStatus.ACTIVE)
        active_valid = active.any(dim=1)
        active_slot = active.long().argmax(dim=1)
        offset = self.position_w.gather(1, active_slot[:, None, None].expand(-1, -1, 3)).squeeze(1)
        offset = torch.where(active_valid[:, None], offset - robot_position_w, torch.zeros_like(offset))

        distance = torch.linalg.vector_norm(self.position_w - robot_position_w[:, None, :], dim=-1)
        candidate = self.status == int(OpeningStatus.CANDIDATE)
        nearest = torch.where(candidate, distance, torch.full_like(distance, torch.inf)).min(dim=1).values
        nearest = torch.where(torch.isfinite(nearest), nearest, torch.zeros_like(nearest))
        return torch.cat((counts, active_valid.float()[:, None], offset, nearest[:, None]), dim=-1)
