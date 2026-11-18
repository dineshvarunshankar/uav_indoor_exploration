"""Rough opening search state, kept separate from detected openings."""

from __future__ import annotations

from enum import IntEnum

import torch


def project_exterior_station(
    hint_w: torch.Tensor,
    anchor_w: torch.Tensor,
    normal_w: torch.Tensor,
    standoff_m: torch.Tensor,
    floor_z: torch.Tensor,
) -> torch.Tensor:
    """Project a rough hint onto a station outside the current facade."""

    if hint_w.ndim != 2 or hint_w.shape[-1] != 3 or anchor_w.shape != hint_w.shape:
        raise ValueError("hint_w and anchor_w must have shape (environments, 3)")
    if normal_w.shape != hint_w.shape:
        raise ValueError("normal_w must match hint_w")
    if standoff_m.shape != hint_w.shape[:1] or floor_z.shape != hint_w.shape[:1]:
        raise ValueError("standoff_m and floor_z must contain one value per environment")
    if not all(torch.isfinite(value).all() for value in (hint_w, anchor_w, normal_w, standoff_m, floor_z)):
        raise ValueError("search station inputs must be finite")
    length = normal_w.norm(dim=-1)
    if torch.any(length <= 1.0e-6):
        raise ValueError("facade normal must be non-zero")
    normal = normal_w / length[:, None]
    tangent = torch.stack((-normal[:, 1], normal[:, 0], torch.zeros_like(normal[:, 0])), dim=-1)
    along = ((hint_w - anchor_w) * tangent).sum(dim=-1)
    wall_point = anchor_w + tangent * along[:, None]
    wall_point[:, 2] = floor_z
    return wall_point + normal * standoff_m[:, None]


class ProposalSearchStatus(IntEnum):
    """Life cycle of one rough opening proposal."""

    PENDING = 0
    SEARCHING = 1
    MATCHED = 2
    INSPECTED = 3
    UNSAFE = 4
    UNRESOLVED = 5


class ProposalSearchRegistry:
    """Per-environment search results for one shared proposal ID set."""

    def __init__(self, num_envs: int, device: torch.device | str) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.ids: tuple[str, ...] = ()
        self.position_local = torch.empty(num_envs, 0, 3, device=self.device)
        self.status = torch.empty(num_envs, 0, dtype=torch.long, device=self.device)
        self.matched_slot = torch.empty(num_envs, 0, dtype=torch.long, device=self.device)
        self.valid = torch.empty(num_envs, 0, dtype=torch.bool, device=self.device)
        self.map_version = torch.empty(num_envs, 0, dtype=torch.long, device=self.device)

    @property
    def proposal_count(self) -> int:
        return len(self.ids)

    def _environment_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        ids = env_ids.to(device=self.device, dtype=torch.long)
        if ids.ndim != 1 or torch.any(ids < 0) or torch.any(ids >= self.num_envs):
            raise ValueError("env_ids must contain valid environment indices")
        return ids

    def replace(
        self,
        ids: tuple[str, ...],
        position_local: torch.Tensor,
        map_version: int,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Replace the complete proposal map for selected environments."""

        if len(ids) != len(set(ids)):
            raise ValueError("duplicate proposal id")
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError("proposal ids must be non-empty strings")
        selected = self._environment_ids(env_ids)
        expected = (selected.numel(), len(ids), 3)
        if tuple(position_local.shape) != expected:
            raise ValueError(f"position_local shape must be {expected}")
        if not torch.isfinite(position_local).all():
            raise ValueError("proposal positions must be finite")
        if map_version < 0:
            raise ValueError("map_version must be non-negative")

        if env_ids is None:
            self.ids = tuple(ids)
            self.position_local = position_local.to(self.device).clone()
            shape = (self.num_envs, len(ids))
            self.status = torch.full(
                shape, int(ProposalSearchStatus.PENDING), dtype=torch.long, device=self.device
            )
            self.matched_slot = torch.full(shape, -1, dtype=torch.long, device=self.device)
            self.valid = torch.ones(shape, dtype=torch.bool, device=self.device)
            self.map_version = torch.full(shape, map_version, dtype=torch.long, device=self.device)
            return

        if tuple(ids) != self.ids:
            raise ValueError("partial replacement must keep the shared proposal ids")
        self.position_local[selected] = position_local.to(self.device)
        self.status[selected] = int(ProposalSearchStatus.PENDING)
        self.matched_slot[selected] = -1
        self.valid[selected] = True
        self.map_version[selected] = map_version

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset search results without changing the accepted proposal map."""

        selected = self._environment_ids(env_ids)
        self.status[selected] = torch.where(
            self.valid[selected],
            torch.full_like(self.status[selected], int(ProposalSearchStatus.PENDING)),
            self.status[selected],
        )
        self.matched_slot[selected] = -1

    def mark(
        self,
        slot: torch.Tensor,
        status: ProposalSearchStatus,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Set one proposal status for each selected environment."""

        slots = slot.to(device=self.device, dtype=torch.long)
        if slots.shape != (self.num_envs,):
            raise ValueError("slot must contain one index per environment")
        selected = self.valid & (
            torch.arange(self.proposal_count, device=self.device)[None, :] == slots[:, None]
        )
        if mask is not None:
            selected &= mask.to(device=self.device, dtype=torch.bool)[:, None]
        self.status = torch.where(selected, torch.full_like(self.status, int(status)), self.status)

    def associate(
        self,
        position_w: torch.Tensor,
        slot: torch.Tensor,
        eligible: torch.Tensor,
        maximum_association_residual_m: float | None,
        searchable_hint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Match only unambiguous detector positions inside an explicit residual."""

        if position_w.ndim != 3 or position_w.shape[0] != self.num_envs or position_w.shape[-1] != 3:
            raise ValueError("position_w must have shape (environments, detections, 3)")
        detections = position_w.shape[1]
        if slot.shape != (self.num_envs, detections) or eligible.shape != slot.shape:
            raise ValueError("slot and eligible must match the detection shape")
        if searchable_hint is not None and searchable_hint.shape != self.valid.shape:
            raise ValueError("searchable_hint must match proposal state shape")
        matched = torch.zeros_like(self.valid)
        if maximum_association_residual_m is None or self.proposal_count == 0 or detections == 0:
            return matched
        residual = torch.as_tensor(maximum_association_residual_m, dtype=torch.float32)
        if not torch.isfinite(residual) or residual <= 0.0:
            raise ValueError("maximum_association_residual_m must be finite and positive")
        if not torch.isfinite(position_w).all():
            raise ValueError("detector positions must be finite")

        searchable = self.valid & (
            (self.status == int(ProposalSearchStatus.PENDING))
            | (self.status == int(ProposalSearchStatus.SEARCHING))
        )
        if searchable_hint is not None:
            searchable &= searchable_hint.to(device=self.device, dtype=torch.bool)
        distance = torch.cdist(self.position_local, position_w.to(self.device))
        inside = (
            searchable[:, :, None]
            & eligible.to(device=self.device, dtype=torch.bool)[:, None, :]
            & (distance <= maximum_association_residual_m)
        )
        unique_for_hint = inside.sum(dim=2) == 1
        unique_for_detection = inside.sum(dim=1) == 1
        pairs = inside & unique_for_hint[:, :, None] & unique_for_detection[:, None, :]
        matched = pairs.any(dim=2)
        detection_index = pairs.long().argmax(dim=2)
        selected_slot = slot.to(self.device).gather(1, detection_index)
        self.matched_slot = torch.where(matched, selected_slot, self.matched_slot)
        self.status = torch.where(
            matched,
            torch.full_like(self.status, int(ProposalSearchStatus.MATCHED)),
            self.status,
        )
        return matched

    def sync_terminal(
        self,
        ledger_status: torch.Tensor,
        inspected_status: int,
        unsafe_status: int,
    ) -> None:
        """Copy terminal results from each matched detector-ledger slot."""

        if ledger_status.ndim != 2 or ledger_status.shape[0] != self.num_envs:
            raise ValueError("ledger_status must have shape (environments, slots)")
        if ledger_status.shape[1] == 0 or self.proposal_count == 0:
            return
        valid_slot = (self.matched_slot >= 0) & (self.matched_slot < ledger_status.shape[1])
        safe_slot = self.matched_slot.clamp(0, ledger_status.shape[1] - 1)
        gathered = ledger_status.to(self.device).gather(1, safe_slot)
        matched = valid_slot & (self.status == int(ProposalSearchStatus.MATCHED))
        self.status = torch.where(
            matched & (gathered == inspected_status),
            torch.full_like(self.status, int(ProposalSearchStatus.INSPECTED)),
            self.status,
        )
        self.status = torch.where(
            matched & (gathered == unsafe_status),
            torch.full_like(self.status, int(ProposalSearchStatus.UNSAFE)),
            self.status,
        )

    def select_pending(
        self,
        reference_w: torch.Tensor,
        environment_floor_z: torch.Tensor,
        floor_band_height_m: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the nearest pending hint assigned to the current floor band."""

        if reference_w.shape != (self.num_envs, 3):
            raise ValueError("reference_w must have shape (environments, 3)")
        if environment_floor_z.shape != (self.num_envs,):
            raise ValueError("environment_floor_z must contain one value per environment")
        if not torch.isfinite(torch.as_tensor(floor_band_height_m)) or floor_band_height_m <= 0.0:
            raise ValueError("floor_band_height_m must be finite and positive")
        if self.proposal_count == 0:
            return (
                torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            )
        pending = self.valid & (self.status == int(ProposalSearchStatus.PENDING))
        on_floor = (self.position_local[..., 2] - environment_floor_z[:, None]).abs() <= (
            0.5 * floor_band_height_m
        )
        distance = torch.linalg.vector_norm(
            self.position_local - reference_w.to(self.device)[:, None, :], dim=-1
        )
        ranked = torch.where(pending & on_floor, distance, torch.full_like(distance, torch.inf))
        best = ranked.min(dim=1)
        return best.indices, torch.isfinite(best.values)

    def mark_remaining_unresolved(self, mask: torch.Tensor) -> None:
        """Mark all unverified hints when the complete search ends."""

        environments = mask.to(device=self.device, dtype=torch.bool)[:, None]
        unfinished = (self.status == int(ProposalSearchStatus.PENDING)) | (
            self.status == int(ProposalSearchStatus.SEARCHING)
        )
        selected = environments & self.valid & unfinished
        self.status = torch.where(
            selected,
            torch.full_like(self.status, int(ProposalSearchStatus.UNRESOLVED)),
            self.status,
        )

    @property
    def unresolved_count(self) -> torch.Tensor:
        return ((self.status == int(ProposalSearchStatus.UNRESOLVED)) & self.valid).sum(dim=1)

    @property
    def success_ready(self) -> torch.Tensor:
        terminal = (self.status == int(ProposalSearchStatus.INSPECTED)) | (
            self.status == int(ProposalSearchStatus.UNSAFE)
        )
        return (~self.valid | terminal).all(dim=1)
