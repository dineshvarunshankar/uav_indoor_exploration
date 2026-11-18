"""Fixed-capacity opening inventory: the long-term building record."""

from __future__ import annotations

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


# Mutable Hydra configuration.
@dataclass
class LedgerCfg:
    """Provisional capacity and matching thresholds."""

    # Opening capacity.
    max_openings: int = 48
    merge_radius_m: float = 1.5
    # Minimum observations for approach.
    min_hits_for_candidate: int = 3
    association_gate_m: float = 1.2
    facade_membership_m: float = 1.5  # maximum facade distance, in meters
    solved_evidence_bonus: float = 1.0  # confirmed-candidate bonus
    # Unsolved candidate lifetime.
    max_unsolved_age_s: float = 30.0
    max_insertions_per_step: int = 4
    image_merge_gate: float = 0.12
    # Maximum best-to-second-best distance ratio.
    identity_ambiguity_ratio: float = 0.5


class OpeningLedger:
    """Batched opening inventory, one entry set per environment."""

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
        self.box_xywhn = torch.zeros(num_envs, slots, 4, device=self.device)
        self.track_id = torch.full((num_envs, slots), -1, dtype=torch.long, device=self.device)
        # Observed station-to-opening direction.
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

        # Triangulation normal equations: twelve values per slot.
        self.ray_a = torch.zeros(num_envs, slots, 3, 3, device=self.device)
        self.ray_b = torch.zeros(num_envs, slots, 3, device=self.device)
        self.solved = torch.zeros(num_envs, slots, dtype=torch.bool, device=self.device)
        self.conditioning = torch.zeros(num_envs, slots, device=self.device)
        # Validated clearance, separate from inspection status.
        self.validated = torch.zeros(num_envs, slots, dtype=torch.bool, device=self.device)
        self.stale_pruned_total = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.replaced_total = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.off_facade_pruned_total = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.refresh_attempted = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.refresh_no_candidate = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.refresh_ambiguous = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

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
        self.box_xywhn[env_ids] = 0.0
        self.track_id[env_ids] = -1
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
        self.ray_a[env_ids] = 0.0
        self.ray_b[env_ids] = 0.0
        self.solved[env_ids] = False
        self.conditioning[env_ids] = 0.0
        self.validated[env_ids] = False
        self.stale_pruned_total[env_ids] = 0
        self.replaced_total[env_ids] = 0
        self.off_facade_pruned_total[env_ids] = 0

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

        # Preserve terminal entries.
        terminal_match = matched & self.terminal.gather(1, matched_slot)
        merged = matched & ~terminal_match
        self._merge(matched_slot, merged, position_w, approach_yaw, size_wh, confidence, time_s)

        unplaced = self._insert_free(
            valid & ~matched, position_w, approach_yaw, size_wh, floor, confidence, time_s, source, map_version
        )
        self._replace_weakest(
            unplaced, position_w, approach_yaw, size_wh, floor, confidence, time_s, source, map_version
        )

    def observe_bearings(
        self,
        origin_w: torch.Tensor,
        ray_w: torch.Tensor,
        seen_w: torch.Tensor,
        seen_valid: torch.Tensor,
        confidence: torch.Tensor,
        floor: torch.Tensor,
        time_s: float,
    ) -> None:
        """Accumulate bearings; seen_w supplies world-position seeds for association."""

        self.prune_stale_unsolved(time_s)
        slot_of = self._assign_bearings(seen_w, seen_valid, floor)
        matched = slot_of >= 0
        safe_slot = slot_of.clamp_min(0)

        # Preserve terminal matches without accumulation or insertion.
        accumulate = matched & ~self.terminal.gather(1, safe_slot)

        self._accumulate(safe_slot, accumulate, origin_w, ray_w, confidence, time_s)
        self._insert_bearings(seen_w, seen_valid & ~matched, confidence, floor, time_s)
        self._solve_positions()

    def observe_tracks(
        self,
        position_w: torch.Tensor,
        approach_yaw: torch.Tensor,
        size_wh: torch.Tensor,
        floor: torch.Tensor,
        track_id: torch.Tensor,
        box_xywhn: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        time_s: float,
    ) -> None:
        """Merge confirmed image tracks into the long-term opening inventory."""

        slot_of = self._assign_tracks(position_w, box_xywhn, track_id, floor, valid)
        matched = slot_of >= 0
        safe_slot = slot_of.clamp_min(0)
        terminal_match = matched & self.terminal.gather(1, safe_slot)
        merged = matched & ~terminal_match
        self._merge_tracks(
            safe_slot,
            merged,
            position_w,
            approach_yaw,
            size_wh,
            floor,
            track_id,
            box_xywhn,
            confidence,
            time_s,
        )
        self._insert_tracks(
            valid & ~matched,
            position_w,
            approach_yaw,
            size_wh,
            floor,
            track_id,
            box_xywhn,
            confidence,
            time_s,
        )

    def refresh_active(
        self,
        slot: torch.Tensor,
        active: torch.Tensor,
        track_id: torch.Tensor,
        box_xywhn: torch.Tensor,
        position_w: torch.Tensor,
        valid: torch.Tensor,
        time_s: float,
    ) -> torch.Tensor:
        """Keep the active entry's image identity current after the survey station."""

        index = slot.clamp_min(0)[:, None]
        stored_id = self.track_id.gather(1, index)[:, 0]
        stored_box = self.box_xywhn.gather(1, index[..., None].expand(-1, -1, 4))[:, 0]
        stored_position = self.position_w.gather(1, index[..., None].expand(-1, -1, 3))[:, 0]

        identity = valid & (stored_id[:, None] >= 0) & (track_id == stored_id[:, None])
        box_gap = torch.linalg.vector_norm(box_xywhn[..., :2] - stored_box[:, None, :2], dim=-1)
        world_gap = torch.linalg.vector_norm(position_w - stored_position[:, None, :], dim=-1)
        # Bearing-only entries have no stored box.
        box_known = stored_box.abs().sum(dim=-1) > 0.0
        # Require an unambiguous world-position match.
        near_world = valid & (world_gap <= self.cfg.association_gate_m)
        ranked_world = torch.where(near_world, world_gap, torch.full_like(world_gap, torch.inf))
        ordered = ranked_world.sort(dim=1).values
        runner_up = ordered[:, 1] if ordered.shape[1] > 1 else torch.full_like(ordered[:, 0], torch.inf)
        unambiguous = ordered[:, 0] <= self.cfg.identity_ambiguity_ratio * runner_up
        eligible = valid & (
            identity
            | (box_known[:, None] & (box_gap <= self.cfg.image_merge_gate))
            | (near_world & unambiguous[:, None])
        )
        # Prioritize matching track IDs.
        rank = torch.where(identity, torch.full_like(box_gap, -1.0), box_gap)
        rank = torch.where(eligible, rank, torch.full_like(rank, torch.inf))
        best = rank.argmin(dim=1)
        found = active & (slot >= 0) & torch.isfinite(rank.gather(1, best[:, None])[:, 0])

        # Refresh failure counters.
        attempted = active & (slot >= 0) & valid.any(dim=1)
        self.refresh_attempted = attempted
        self.refresh_no_candidate = attempted & ~eligible.any(dim=1)
        self.refresh_ambiguous = attempted & near_world.any(dim=1) & ~unambiguous & ~eligible.any(dim=1)

        rows = torch.arange(self.num_envs, device=self.device)[found]
        if rows.numel() > 0:
            columns = slot[found]
            picked = best[found]
            self.track_id[rows, columns] = track_id[rows, picked]
            self.box_xywhn[rows, columns] = box_xywhn[rows, picked]
            self.last_seen_s[rows, columns] = time_s
        return found

    def _assign_tracks(
        self,
        position_w: torch.Tensor,
        box_xywhn: torch.Tensor,
        track_id: torch.Tensor,
        floor: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Assign image tracks to existing non-empty slots, one detection per slot."""

        detections = box_xywhn.shape[1]
        distance = torch.cdist(self.box_xywhn[..., :2], box_xywhn[..., :2])
        same_floor = self.floor[:, :, None] == floor[:, None, :]
        same_track = (
            self.occupied[:, :, None]
            & same_floor
            & valid[:, None, :]
            & (self.track_id[:, :, None] >= 0)
            & (self.track_id[:, :, None] == track_id[:, None, :])
        )
        # Require world-position agreement for a new track ID.
        world_distance = torch.cdist(self.position_w, position_w)
        nearby_box = (
            self.occupied[:, :, None]
            & same_floor
            & valid[:, None, :]
            & (distance <= self.cfg.image_merge_gate)
            & (world_distance <= self.cfg.association_gate_m)
        )
        score = torch.where(
            same_track,
            distance * 0.01,
            torch.where(nearby_box, distance, torch.full_like(distance, torch.inf)),
        )

        envs, slots, _ = score.shape
        slot_of = torch.full((envs, detections), -1, dtype=torch.long, device=self.device)
        env_index = torch.arange(envs, device=self.device)
        for _ in range(min(slots, detections)):
            flat = score.flatten(1).min(dim=1)
            found = torch.isfinite(flat.values)
            if not bool(found.any()):
                break
            slot = torch.div(flat.indices, detections, rounding_mode="floor")
            detection = flat.indices % detections
            slot_of[env_index[found], detection[found]] = slot[found]
            score[env_index, slot, :] = torch.inf
            score[env_index, :, detection] = torch.inf
        return slot_of

    def _merge_tracks(
        self,
        slot: torch.Tensor,
        merged: torch.Tensor,
        position_w: torch.Tensor,
        approach_yaw: torch.Tensor,
        size_wh: torch.Tensor,
        floor: torch.Tensor,
        track_id: torch.Tensor,
        box_xywhn: torch.Tensor,
        confidence: torch.Tensor,
        time_s: float,
    ) -> None:
        env_index = torch.arange(self.num_envs, device=self.device)
        for detection in range(merged.shape[1]):
            take = merged[:, detection]
            if not bool(take.any()):
                continue
            rows = env_index[take]
            columns = slot[take, detection]
            weight = confidence[take, detection]
            total = (self.weight[rows, columns] + weight).clamp_min(1.0e-6)
            self.position_w[rows, columns] = (
                self.position_w[rows, columns] * self.weight[rows, columns, None]
                + position_w[take, detection] * weight[:, None]
            ) / total[:, None]
            self.approach_yaw[rows, columns] = (
                self.approach_yaw[rows, columns] * self.weight[rows, columns]
                + approach_yaw[take, detection] * weight
            ) / total
            self.size_wh[rows, columns] = (
                self.size_wh[rows, columns] * self.weight[rows, columns, None]
                + size_wh[take, detection] * weight[:, None]
            ) / total[:, None]
            self.weight[rows, columns] = total
            self.confidence[rows, columns] = torch.maximum(self.confidence[rows, columns], weight)
            self.hits[rows, columns] += 1
            self.floor[rows, columns] = floor[take, detection]
            self.track_id[rows, columns] = track_id[take, detection]
            self.box_xywhn[rows, columns] = box_xywhn[take, detection]
            self.last_seen_s[rows, columns] = time_s

    def _insert_tracks(
        self,
        unplaced: torch.Tensor,
        position_w: torch.Tensor,
        approach_yaw: torch.Tensor,
        size_wh: torch.Tensor,
        floor: torch.Tensor,
        track_id: torch.Tensor,
        box_xywhn: torch.Tensor,
        confidence: torch.Tensor,
        time_s: float,
    ) -> None:
        detections = unplaced.shape[1]
        order = torch.argsort(
            torch.where(unplaced, confidence, torch.full_like(confidence, -torch.inf)),
            dim=1,
            descending=True,
        )
        env_index = torch.arange(self.num_envs, device=self.device)
        for rank in range(min(self.cfg.max_insertions_per_step, detections)):
            detection = order[:, rank]
            take = unplaced.gather(1, detection[:, None]).squeeze(1)
            free = ~self.occupied
            has_free = free.any(dim=1)
            free_slot = torch.argmax(free.float(), dim=1)

            candidate = self.status == int(OpeningStatus.CANDIDATE)
            confirmed = self.hits >= self.cfg.min_hits_for_candidate
            evidence = self.confidence + confirmed.float() * self.cfg.solved_evidence_bonus
            held = torch.where(candidate, evidence, torch.full_like(evidence, torch.inf))
            weakest = held.min(dim=1)
            picked_confidence = confidence.gather(1, detection[:, None]).squeeze(1)
            replace = ~has_free & (picked_confidence > weakest.values)
            slot = torch.where(has_free, free_slot, weakest.indices)
            take = take & (has_free | replace)
            if not bool(take.any()):
                break
            rows = env_index[take]
            columns = slot[take]
            picked = detection[take]
            self.replaced_total[rows] += replace[take].long()
            self.position_w[rows, columns] = position_w[rows, picked]
            self.approach_yaw[rows, columns] = approach_yaw[rows, picked]
            self.size_wh[rows, columns] = size_wh[rows, picked]
            self.floor[rows, columns] = floor[rows, picked]
            self.track_id[rows, columns] = track_id[rows, picked]
            self.box_xywhn[rows, columns] = box_xywhn[rows, picked]
            self.confidence[rows, columns] = confidence[rows, picked]
            self.weight[rows, columns] = confidence[rows, picked]
            self.status[rows, columns] = int(OpeningStatus.CANDIDATE)
            self.hits[rows, columns] = 1
            self.solved[rows, columns] = False
            self.conditioning[rows, columns] = 0.0
            self.validated[rows, columns] = False
            self.ray_a[rows, columns] = 0.0
            self.ray_b[rows, columns] = 0.0
            self.last_seen_s[rows, columns] = time_s
            self.source[rows, columns] = 0
            self.map_version[rows, columns] = 0
            self.opening_id[rows, columns] = self._next_id[rows]
            self._next_id[rows] = self._next_id[rows] + 1

    def _solve_positions(self, min_conditioning: float = 0.008) -> None:
        """Solve each slot's normal equations, keeping only well-conditioned answers."""

        # Conditioning from singular values.
        spectrum = torch.linalg.svdvals(self.ray_a)
        ratio = spectrum[..., -1] / spectrum[..., 0].clamp_min(1e-9)
        conditioned = ratio > min_conditioning
        regularised = self.ray_a + torch.eye(3, device=self.device) * 1e-6
        position = (torch.linalg.pinv(regularised) @ self.ray_b.unsqueeze(-1)).squeeze(-1)

        usable = conditioned & self.occupied & (self.hits > 0)
        self.position_w = torch.where(usable[..., None], position, self.position_w)
        self.conditioning = torch.where(self.occupied, ratio, self.conditioning)
        self.solved = self.solved | usable

    def _assign_bearings(
        self, seen_w: torch.Tensor, seen_valid: torch.Tensor, floor: torch.Tensor
    ) -> torch.Tensor:
        """Assign detections to slots with one-to-one matching."""

        distance = torch.cdist(self.position_w, seen_w)
        same_floor = self.floor[:, :, None] == floor[:, None, :]
        feasible = (
            self.occupied[:, :, None]
            & same_floor
            & seen_valid[:, None, :]
            & (distance <= self.cfg.association_gate_m)
        )
        work = torch.where(feasible, distance, torch.full_like(distance, float("inf")))

        envs, slots, detections = work.shape
        slot_of = torch.full((envs, detections), -1, dtype=torch.long, device=self.device)
        env_index = torch.arange(envs, device=self.device)
        for _ in range(min(slots, detections)):
            flat = work.flatten(1).min(dim=1)
            found = torch.isfinite(flat.values)
            if not bool(found.any()):
                break
            slot = torch.div(flat.indices, detections, rounding_mode="floor")
            detection = flat.indices % detections
            slot_of[env_index[found], detection[found]] = slot[found]
            work[env_index, slot, :] = float("inf")
            work[env_index, :, detection] = float("inf")
        return slot_of

    def _accumulate(
        self,
        slot: torch.Tensor,
        matched: torch.Tensor,
        origin_w: torch.Tensor,
        ray_w: torch.Tensor,
        confidence: torch.Tensor,
        time_s: float,
    ) -> None:
        """Add each matched bearing to its slot's running least-squares state."""

        # Replace invalid rays before accumulation.
        usable = matched & torch.isfinite(ray_w).all(dim=-1)
        default = torch.zeros_like(ray_w)
        default[..., 0] = 1.0
        ray_w = torch.where(usable.unsqueeze(-1), ray_w, default)
        weight = torch.where(usable, confidence, torch.zeros_like(confidence))
        projector = torch.eye(3, device=self.device).expand(*ray_w.shape[:-1], 3, 3)
        projector = (projector - ray_w.unsqueeze(-1) * ray_w.unsqueeze(-2)) * weight[..., None, None]
        contribution = (projector @ origin_w[:, None, :, None].expand(*projector.shape[:-1], 1)).squeeze(-1)

        index_a = slot[..., None, None].expand(-1, -1, 3, 3)
        self.ray_a = self.ray_a.scatter_add(1, index_a, projector)
        self.ray_b = self.ray_b.scatter_add(1, slot[..., None].expand(-1, -1, 3), contribution)

        added_hits = torch.zeros_like(self.hits).scatter_add(1, slot, usable.long())
        self.hits = self.hits + added_hits
        best = torch.zeros_like(self.confidence).scatter_reduce(
            1, slot, weight, reduce="amax", include_self=False
        )
        touched = added_hits > 0
        self.confidence = torch.where(touched, torch.maximum(self.confidence, best), self.confidence)
        self.weight = self.weight + torch.zeros_like(self.weight).scatter_add(1, slot, weight)
        self.last_seen_s = torch.where(touched, torch.full_like(self.last_seen_s, time_s), self.last_seen_s)

    def _insert_bearings(
        self,
        seen_w: torch.Tensor,
        unmatched: torch.Tensor,
        confidence: torch.Tensor,
        floor: torch.Tensor,
        time_s: float,
    ) -> None:
        """Open a slot for each unexplained detection, strongest first."""

        ranked = torch.where(unmatched, confidence, torch.full_like(confidence, -1.0))
        order = torch.argsort(ranked, dim=1, descending=True)
        env_index = torch.arange(self.num_envs, device=self.device)
        for rank in range(min(self.cfg.max_insertions_per_step, seen_w.shape[1])):
            detection = order[:, rank]
            take = unmatched.gather(1, detection[:, None]).squeeze(1)
            free = ~self.occupied
            has_free = free.any(dim=1)
            free_slot = torch.argmax(free.float(), dim=1)

            # Protect solved, active, and terminal entries from replacement.
            replaceable = (self.status == int(OpeningStatus.CANDIDATE)) & ~self.solved
            held = torch.where(replaceable, self.confidence, torch.full_like(self.confidence, torch.inf))
            weakest = held.min(dim=1)
            picked_confidence = confidence.gather(1, detection[:, None]).squeeze(1)
            replace = ~has_free & (picked_confidence > weakest.values)
            slot = torch.where(has_free, free_slot, weakest.indices)
            take = take & (has_free | replace)
            if not bool(take.any()):
                break
            rows = env_index[take]
            columns = slot[take]
            picked = detection[take]
            self.replaced_total[rows] += replace[take].long()
            self.position_w[rows, columns] = seen_w[rows, picked]
            self.box_xywhn[rows, columns] = 0.0
            self.track_id[rows, columns] = -1
            self.floor[rows, columns] = floor[rows, picked]
            self.confidence[rows, columns] = confidence[rows, picked]
            self.status[rows, columns] = int(OpeningStatus.CANDIDATE)
            self.hits[rows, columns] = 0
            self.weight[rows, columns] = 0.0
            self.solved[rows, columns] = False
            self.conditioning[rows, columns] = 0.0
            self.validated[rows, columns] = False
            self.ray_a[rows, columns] = 0.0
            self.ray_b[rows, columns] = 0.0
            self.last_seen_s[rows, columns] = time_s
            self.opening_id[rows, columns] = self._next_id[rows]
            self._next_id[rows] = self._next_id[rows] + 1

    def _clear_slots(self, clear: torch.Tensor) -> None:
        """Return selected entries to EMPTY without reusing their stable identifiers."""

        self.position_w = torch.where(clear[..., None], torch.zeros_like(self.position_w), self.position_w)
        self.box_xywhn = torch.where(clear[..., None], torch.zeros_like(self.box_xywhn), self.box_xywhn)
        self.track_id = torch.where(clear, torch.full_like(self.track_id, -1), self.track_id)
        self.approach_yaw = torch.where(clear, torch.zeros_like(self.approach_yaw), self.approach_yaw)
        self.size_wh = torch.where(clear[..., None], torch.zeros_like(self.size_wh), self.size_wh)
        self.floor = torch.where(clear, torch.zeros_like(self.floor), self.floor)
        self.status = torch.where(clear, torch.full_like(self.status, int(OpeningStatus.EMPTY)), self.status)
        self.confidence = torch.where(clear, torch.zeros_like(self.confidence), self.confidence)
        self.weight = torch.where(clear, torch.zeros_like(self.weight), self.weight)
        self.hits = torch.where(clear, torch.zeros_like(self.hits), self.hits)
        self.last_seen_s = torch.where(clear, torch.zeros_like(self.last_seen_s), self.last_seen_s)
        self.source = torch.where(clear, torch.zeros_like(self.source), self.source)
        self.map_version = torch.where(clear, torch.zeros_like(self.map_version), self.map_version)
        self.opening_id = torch.where(clear, torch.full_like(self.opening_id, -1), self.opening_id)
        self.ray_a = torch.where(clear[..., None, None], torch.zeros_like(self.ray_a), self.ray_a)
        self.ray_b = torch.where(clear[..., None], torch.zeros_like(self.ray_b), self.ray_b)
        self.solved = self.solved & ~clear
        self.conditioning = torch.where(clear, torch.zeros_like(self.conditioning), self.conditioning)
        self.validated = self.validated & ~clear

    def prune_stale_unsolved(self, time_s: float) -> None:
        """Release provisional bearing slots that stopped receiving supporting looks."""

        stale = (time_s - self.last_seen_s) > self.cfg.max_unsolved_age_s
        provisional = (self.status == int(OpeningStatus.CANDIDATE)) & ~self.solved
        clear = provisional & stale
        self.stale_pruned_total += clear.sum(dim=1)
        self._clear_slots(clear)

    def prune_off_facade(
        self, facade_normal_w: torch.Tensor, facade_offset_m: torch.Tensor, facade_valid: torch.Tensor
    ) -> None:
        """Remove solved candidates whose parallax fix disagrees with a reliable wall fit."""

        distance = (
            (self.position_w * facade_normal_w[:, None, :]).sum(dim=-1) - facade_offset_m[:, None]
        ).abs()
        off_wall = distance > self.cfg.facade_membership_m
        disposable = (self.status == int(OpeningStatus.CANDIDATE)) & self.solved
        clear = disposable & off_wall & facade_valid[:, None]
        self.off_facade_pruned_total += clear.sum(dim=1)
        self._clear_slots(clear)

    @property
    def approachable(self) -> torch.Tensor:
        """Require sufficient observations and a solved position or image track."""

        image_backed = self.track_id >= 0
        return (self.solved | image_backed) & (self.hits >= self.cfg.min_hits_for_candidate) & self.occupied

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
        self.solved = torch.where(touched, torch.ones_like(self.solved), self.solved)
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
        self.box_xywhn = torch.where(written[..., None], torch.zeros_like(self.box_xywhn), self.box_xywhn)
        self.track_id = torch.where(written, torch.full_like(self.track_id, -1), self.track_id)
        self.size_wh = torch.where(written[..., None], gathered_size, self.size_wh)
        self.approach_yaw = torch.where(written, approach_yaw.gather(1, source_index), self.approach_yaw)
        self.floor = torch.where(written, floor.gather(1, source_index), self.floor)
        self.confidence = torch.where(written, gathered_confidence, self.confidence)
        self.weight = torch.where(written, gathered_confidence, self.weight)
        self.hits = torch.where(written, torch.ones_like(self.hits), self.hits)
        self.solved = torch.where(written, torch.ones_like(self.solved), self.solved)
        self.validated = torch.where(written, torch.zeros_like(self.validated), self.validated)
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
        """Keep the best-evidenced openings when the ledger is full."""

        detections = unplaced.shape[1]
        for _ in range(min(self.max_openings, detections)):
            offered = torch.where(unplaced, confidence, torch.full_like(confidence, -torch.inf))
            best = offered.max(dim=1)
            candidate = self.status == int(OpeningStatus.CANDIDATE)
            # Add the confirmation bonus.
            confirmed = self.hits >= self.cfg.min_hits_for_candidate
            evidence = self.confidence + confirmed.float() * self.cfg.solved_evidence_bonus
            held = torch.where(candidate, evidence, torch.full_like(evidence, torch.inf))
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

    def mark_validated(self, slot: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Record metric corridor clearance without claiming a completed inspection."""

        selected = torch.arange(self.max_openings, device=self.device)[None, :] == slot[:, None]
        if mask is not None:
            selected = selected & mask[:, None]
        self.validated = self.validated | selected

    def select_candidate(
        self,
        station_position_w: torch.Tensor,
        facade_normal_w: torch.Tensor,
        survey_direction: torch.Tensor,
        floor: torch.Tensor,
        facade_offset_m: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select a stable candidate by confidence along the facade sweep."""

        tangent = torch.stack(
            (-facade_normal_w[:, 1], facade_normal_w[:, 0], torch.zeros_like(facade_normal_w[:, 0])),
            dim=-1,
        )
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        offset = self.position_w - station_position_w[:, None, :]
        along = (offset * tangent[:, None, :]).sum(dim=-1) * survey_direction[:, None]
        stable = (
            (self.status == int(OpeningStatus.CANDIDATE))
            & (self.floor == floor[:, None])
            & self.approachable
        )
        if facade_offset_m is not None:
            from_plane = ((self.position_w * facade_normal_w[:, None, :]).sum(dim=-1)
                          - facade_offset_m[:, None]).abs()
            stable = stable & (from_plane <= self.cfg.facade_membership_m)
        # Select the highest-confidence candidate ahead.
        ranked = torch.where(
            stable & (along >= 0.0), self.confidence, torch.full_like(self.confidence, -torch.inf)
        )
        best = ranked.max(dim=1)
        any_ahead = torch.isfinite(best.values)

        # Fall back to stable candidates behind the station.
        behind = torch.where(stable, self.confidence, torch.full_like(self.confidence, -torch.inf))
        nearest = behind.max(dim=1)
        selected = torch.where(any_ahead, best.indices, nearest.indices)
        valid = any_ahead | torch.isfinite(nearest.values)
        return selected, valid

    def floor_complete(self, floor: torch.Tensor) -> torch.Tensor:
        """True when that floor holds no candidate and no active opening."""

        return ~(self.pending & (self.floor == floor[:, None])).any(dim=1)

    def building_complete(self) -> torch.Tensor:
        """True when every recorded opening has a terminal result."""

        return ~self.pending.any(dim=1)

    def observation(self, robot_position_w: torch.Tensor) -> torch.Tensor:
        """Return normalized status counts, active-target validity and offset, and nearest-candidate range."""

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
