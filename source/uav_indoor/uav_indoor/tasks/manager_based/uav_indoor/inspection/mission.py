"""Vectorized deterministic mission supervisor for building-opening inspection.

The supervisor owns mission order, floor coverage, sensor gates, and terminal results.
The learned policy owns local motion only. Entry safety never depends on a detector label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from .ledger import LedgerCfg, OpeningLedger, OpeningStatus
from .observation import ObservationCfg, PrimaryPersonObservation
from .recovery import BreadcrumbBuffer, RecoveryCfg


class MissionStage(IntEnum):
    """Curriculum stage. Each stage adds the next part of the mission."""

    SURVEY_ONLY = 0
    VALIDATE_ONLY = 1
    FULL = 2


class MissionPhase(IntEnum):
    """Phases owned by deterministic logic."""

    TRANSIT = 0
    SURVEY = 1
    APPROACH = 2
    VALIDATE = 3
    ENTRY = 4
    OBSERVE = 5
    EXIT = 6
    RESUME_SURVEY = 7
    NEXT_FLOOR = 8
    DONE = 9


PHASE_COUNT = len(MissionPhase)


@dataclass(frozen=True)
class BuildingMissionCfg:
    """Mission geometry and timing.

    PROVISIONAL. AGENTS.md lists the scan angles, the facade-coverage threshold, and the
    retry limits as open design questions.
    """

    region_radius_m: float = 8.0
    region_entry_fraction: float = 0.85
    stations_per_floor: int = 12
    standoff_m: float = 6.0
    station_tolerance_m: float = 1.0
    scan_dwell_s: float = 2.0
    approach_tolerance_m: float = 0.8
    validation_hits: int = 5
    validation_timeout_s: float = 8.0
    corridor_clearance_m: float = 1.2
    entry_depth_m: float = 0.9
    entry_tolerance_m: float = 0.35
    observe_timeout_s: float = 30.0
    exit_timeout_s: float = 20.0
    floor_band_height_m: float = 3.0
    max_floors: int = 3
    commit_radius_m: float = 10.0
    horizontal_fov_rad: float = math.radians(79.0)


@dataclass
class MissionReference:
    """World-frame reference and phase metadata emitted for local navigation."""

    position_w: torch.Tensor
    yaw_w: torch.Tensor
    phase: torch.Tensor
    entry_authorized: torch.Tensor
    target_valid: torch.Tensor
    detector_enabled: torch.Tensor
    accepts_proposals: torch.Tensor
    observation_progress: torch.Tensor


def wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap radians to [-pi, pi)."""

    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


class BuildingMissionSupervisor:
    """Batch mission state machine for GPS transit, facade survey, entry, and recovery."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cfg: BuildingMissionCfg | None = None,
        ledger_cfg: LedgerCfg | None = None,
        observation_cfg: ObservationCfg | None = None,
        recovery_cfg: RecoveryCfg | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or BuildingMissionCfg()

        self.ledger = OpeningLedger(num_envs, device, ledger_cfg)
        self.person = PrimaryPersonObservation(num_envs, device, observation_cfg)
        self.breadcrumbs = BreadcrumbBuffer(num_envs, device, recovery_cfg)

        self.phase = torch.full((num_envs,), int(MissionPhase.TRANSIT), dtype=torch.long, device=self.device)
        self.phase_time = torch.zeros(num_envs, device=self.device)
        self.floor = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.station_index = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.station_dwell_s = torch.zeros(num_envs, device=self.device)
        self.ring_start_yaw = torch.zeros(num_envs, device=self.device)
        self.survey_direction = torch.ones(num_envs, device=self.device)
        self.building_center_w = torch.zeros(num_envs, 3, device=self.device)
        self.checkpoint_w = torch.zeros(num_envs, 3, device=self.device)
        self.active_slot = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.validation_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.recovery_failed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._entry_authorized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.stage = torch.full((num_envs,), int(MissionStage.FULL), dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None, survey_direction: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.phase[env_ids] = int(MissionPhase.TRANSIT)
        self.phase_time[env_ids] = 0.0
        self.floor[env_ids] = 0
        self.station_index[env_ids] = 0
        self.station_dwell_s[env_ids] = 0.0
        self.ring_start_yaw[env_ids] = 0.0
        self.building_center_w[env_ids] = 0.0
        self.checkpoint_w[env_ids] = 0.0
        self.active_slot[env_ids] = -1
        self.validation_count[env_ids] = 0
        self.recovery_failed[env_ids] = False
        self._entry_authorized[env_ids] = False
        if survey_direction is None:
            survey_direction = torch.where(
                torch.rand(env_ids.shape[0], device=self.device) < 0.5,
                torch.full((env_ids.shape[0],), -1.0, device=self.device),
                torch.ones(env_ids.shape[0], device=self.device),
            )
        self.survey_direction[env_ids] = survey_direction.to(self.device)
        self.ledger.reset(env_ids)
        self.person.reset(env_ids)
        self.breadcrumbs.reset(env_ids)

    def set_stage(self, stage: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Set the curriculum stage for all or selected environments."""

        if env_ids is None:
            self.stage = stage.to(device=self.device, dtype=torch.long)
        else:
            self.stage[env_ids.to(self.device)] = stage.to(device=self.device, dtype=torch.long)

    def _transition(self, mask: torch.Tensor, phase: MissionPhase) -> None:
        self.phase = torch.where(mask, torch.full_like(self.phase, int(phase)), self.phase)
        self.phase_time = torch.where(mask, torch.zeros_like(self.phase_time), self.phase_time)

    def _in(self, phase: MissionPhase) -> torch.Tensor:
        return self.phase == int(phase)

    @property
    def floor_height_w(self) -> torch.Tensor:
        """Height of the current floor band above the building base."""

        return self.building_center_w[:, 2] + self.cfg.floor_band_height_m * (self.floor.float() + 0.5)

    def station_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the position and facade-facing yaw of the current scan station."""

        step = 2.0 * math.pi / self.cfg.stations_per_floor
        angle = self.ring_start_yaw + self.survey_direction * self.station_index.float() * step
        position = self.building_center_w.clone()
        position[:, 0] = position[:, 0] + self.cfg.standoff_m * torch.cos(angle)
        position[:, 1] = position[:, 1] + self.cfg.standoff_m * torch.sin(angle)
        position[:, 2] = self.floor_height_w
        return position, wrap_pi(angle + math.pi)

    def _active(self, values: torch.Tensor, default: float = 0.0) -> torch.Tensor:
        """Gather one ledger field for the active opening, or a default when none is active."""

        index = self.active_slot.clamp_min(0)
        if values.dim() == 3:
            gathered = values.gather(1, index[:, None, None].expand(-1, -1, values.shape[-1])).squeeze(1)
            return torch.where((self.active_slot >= 0)[:, None], gathered, torch.full_like(gathered, default))
        gathered = values.gather(1, index[:, None]).squeeze(1)
        return torch.where(self.active_slot >= 0, gathered, torch.full_like(gathered, default))

    def update(
        self,
        dt: float,
        robot_position_w: torch.Tensor,
        robot_yaw_w: torch.Tensor,
        gps_target_w: torch.Tensor,
        opening_position_w: torch.Tensor,
        opening_approach_yaw: torch.Tensor,
        opening_size_wh: torch.Tensor,
        opening_confidence: torch.Tensor,
        opening_valid: torch.Tensor,
        person_track_id: torch.Tensor,
        person_box_xywhn: torch.Tensor,
        person_confidence: torch.Tensor,
        person_age_s: torch.Tensor,
        person_valid: torch.Tensor,
        tof_valid: torch.Tensor,
        tof_range_m: torch.Tensor,
        clearance_m: torch.Tensor,
        avoidance_active: torch.Tensor,
        time_s: float,
    ) -> MissionReference:
        """Advance mission state and return the current local-motion reference."""

        self.phase_time = self.phase_time + dt

        # TRANSIT: fly to the supplied GPS area, then anchor the building and open a ring.
        transit = self._in(MissionPhase.TRANSIT)
        at_region = (
            torch.linalg.vector_norm(gps_target_w[:, :2] - robot_position_w[:, :2], dim=-1)
            <= self.cfg.region_radius_m
        )
        arrived = transit & at_region
        self.building_center_w = torch.where(arrived[:, None], gps_target_w, self.building_center_w)
        # The ring starts on the side the aircraft came from: that side is known to be outside.
        approach_side = torch.atan2(
            robot_position_w[:, 1] - gps_target_w[:, 1], robot_position_w[:, 0] - gps_target_w[:, 0]
        )
        self.ring_start_yaw = torch.where(transit & ~at_region, approach_side, self.ring_start_yaw)
        self.station_index = torch.where(arrived, torch.zeros_like(self.station_index), self.station_index)
        self.station_dwell_s = torch.where(arrived, torch.zeros_like(self.station_dwell_s), self.station_dwell_s)
        self._transition(arrived, MissionPhase.SURVEY)

        # SURVEY: stop, scan the facade, then move to the next station.
        survey = self._in(MissionPhase.SURVEY)
        station_position, station_yaw = self.station_pose()
        at_station = (
            torch.linalg.vector_norm(robot_position_w - station_position, dim=-1) <= self.cfg.station_tolerance_m
        )
        scanning = survey & at_station
        self.station_dwell_s = torch.where(scanning, self.station_dwell_s + dt, self.station_dwell_s)
        self.ledger.observe(
            opening_position_w,
            opening_approach_yaw,
            opening_size_wh,
            self.floor[:, None].expand_as(opening_confidence),
            opening_confidence,
            opening_valid & scanning[:, None],
            time_s,
        )

        scanned = scanning & (self.station_dwell_s >= self.cfg.scan_dwell_s)
        selected, has_candidate = self.ledger.select_candidate(
            self.building_center_w, station_yaw, self.survey_direction, self.floor
        )
        commit = scanned & has_candidate
        self.active_slot = torch.where(commit, selected, self.active_slot)
        self.checkpoint_w = torch.where(commit[:, None], station_position, self.checkpoint_w)
        self.ledger.mark(selected, OpeningStatus.ACTIVE, commit)
        surveyed = commit & (self.stage == int(MissionStage.SURVEY_ONLY))
        self.ledger.mark(selected, OpeningStatus.INSPECTED, surveyed)
        self._transition(surveyed, MissionPhase.RESUME_SURVEY)
        self._transition(commit & ~surveyed, MissionPhase.APPROACH)

        advance_station = scanned & ~has_candidate
        self.station_index = self.station_index + advance_station.long()
        self.station_dwell_s = torch.where(
            advance_station, torch.zeros_like(self.station_dwell_s), self.station_dwell_s
        )
        ring_done = self._in(MissionPhase.SURVEY) & (self.station_index >= self.cfg.stations_per_floor)
        floor_done = ring_done & self.ledger.floor_complete(self.floor)
        self.station_index = torch.where(
            ring_done & ~floor_done, torch.zeros_like(self.station_index), self.station_index
        )
        self._transition(floor_done, MissionPhase.NEXT_FLOOR)

        # APPROACH and VALIDATE work from the ledger entry, never from a live detection.
        opening_w = self._active(self.ledger.position_w)
        approach_yaw = self._active(self.ledger.approach_yaw)
        # The aircraft waits back along the ray it saw the opening on, and faces down it.
        standoff_position = opening_w.clone()
        standoff_position[:, 0] = standoff_position[:, 0] - self.cfg.standoff_m * torch.cos(approach_yaw)
        standoff_position[:, 1] = standoff_position[:, 1] - self.cfg.standoff_m * torch.sin(approach_yaw)
        facing_yaw = wrap_pi(approach_yaw)

        approach = self._in(MissionPhase.APPROACH)
        at_standoff = torch.linalg.vector_norm(robot_position_w - standoff_position, dim=-1) <= (
            self.cfg.approach_tolerance_m
        )
        start_validation = approach & at_standoff
        self.validation_count = torch.where(
            start_validation, torch.zeros_like(self.validation_count), self.validation_count
        )
        self._transition(start_validation, MissionPhase.VALIDATE)

        # Repeated metric evidence, never an open-vocabulary label, authorizes entry.
        validate = self._in(MissionPhase.VALIDATE)
        evidence = (
            tof_valid
            & (tof_range_m >= self.cfg.corridor_clearance_m)
            & (clearance_m >= self.cfg.corridor_clearance_m)
        )
        cleared = torch.where(validate, torch.zeros_like(self.validation_count), self.validation_count)
        self.validation_count = torch.where(validate & evidence, self.validation_count + 1, cleared)
        authorized = validate & (self.validation_count >= self.cfg.validation_hits)
        validated_only = authorized & (self.stage == int(MissionStage.VALIDATE_ONLY))
        self.ledger.mark(self.active_slot.clamp_min(0), OpeningStatus.INSPECTED, validated_only)
        self._transition(validated_only, MissionPhase.RESUME_SURVEY)
        authorized = authorized & ~validated_only
        self.breadcrumbs.start(authorized)
        self._transition(authorized, MissionPhase.ENTRY)

        rejected = validate & ~authorized & (self.phase_time >= self.cfg.validation_timeout_s)
        self.ledger.mark(self.active_slot.clamp_min(0), OpeningStatus.UNSAFE, rejected)
        self._transition(rejected, MissionPhase.RESUME_SURVEY)

        # ENTRY: cross the authorized corridor and record the protected route.
        entry = self._in(MissionPhase.ENTRY)
        self._entry_authorized = entry
        inside_position = opening_w.clone()
        inside_position[:, 0] = inside_position[:, 0] + self.cfg.entry_depth_m * torch.cos(approach_yaw)
        inside_position[:, 1] = inside_position[:, 1] + self.cfg.entry_depth_m * torch.sin(approach_yaw)
        self.breadcrumbs.push(robot_position_w, robot_yaw_w, clearance_m, time_s, entry)
        inside = entry & (
            torch.linalg.vector_norm(robot_position_w - inside_position, dim=-1) <= self.cfg.entry_tolerance_m
        )
        self.breadcrumbs.seal(inside)
        self._transition(inside, MissionPhase.OBSERVE)

        # OBSERVE: hold position and centre the primary person with yaw alone.
        observe = self._in(MissionPhase.OBSERVE)
        self.person.update(
            dt,
            person_track_id,
            person_box_xywhn,
            person_confidence,
            person_age_s,
            person_valid,
            avoidance_active,
            time_s,
            enabled=observe,
        )
        centring_yaw = wrap_pi(robot_yaw_w - self.person.primary_offset * self.cfg.horizontal_fov_rad)
        observation_done = observe & (self.person.complete | (self.phase_time >= self.cfg.observe_timeout_s))
        self._transition(observation_done, MissionPhase.EXIT)

        # EXIT: fly the saved corridor in reverse while facing forward.
        exit_phase = self._in(MissionPhase.EXIT)
        crumb_position, crumb_yaw, crumb_valid = self.breadcrumbs.next_crumb()
        self.breadcrumbs.advance(robot_position_w, exit_phase)
        outside = exit_phase & self.breadcrumbs.exit_complete
        active_status = self._active(self.ledger.status.float(), float(OpeningStatus.UNSAFE))
        inspected = outside & (active_status != int(OpeningStatus.UNSAFE))
        self.ledger.mark(self.active_slot.clamp_min(0), OpeningStatus.INSPECTED, inspected)
        self._transition(outside, MissionPhase.RESUME_SURVEY)

        stuck = exit_phase & ~outside & (self.phase_time >= self.cfg.exit_timeout_s)
        self.breadcrumbs.mark_failed(stuck)
        self.recovery_failed = self.recovery_failed | stuck
        self._transition(stuck, MissionPhase.DONE)

        # RESUME_SURVEY: return to the near exterior checkpoint, not the distant original.
        resume = self._in(MissionPhase.RESUME_SURVEY)
        back = resume & (
            torch.linalg.vector_norm(robot_position_w - self.checkpoint_w, dim=-1) <= self.cfg.station_tolerance_m
        )
        self.active_slot = torch.where(back, torch.full_like(self.active_slot, -1), self.active_slot)
        self.station_dwell_s = torch.where(back, torch.zeros_like(self.station_dwell_s), self.station_dwell_s)
        self.person.clear(back)
        self._transition(back, MissionPhase.SURVEY)

        # NEXT_FLOOR: open the ring one band higher, or finish the building.
        next_floor = self._in(MissionPhase.NEXT_FLOOR)
        self.floor = self.floor + next_floor.long()
        self.station_index = torch.where(next_floor, torch.zeros_like(self.station_index), self.station_index)
        self.station_dwell_s = torch.where(next_floor, torch.zeros_like(self.station_dwell_s), self.station_dwell_s)
        finished = next_floor & (self.floor >= self.cfg.max_floors)
        self._transition(next_floor & ~finished, MissionPhase.SURVEY)
        self._transition(finished, MissionPhase.DONE)

        return self._reference(
            robot_position_w, robot_yaw_w, gps_target_w, station_position, station_yaw,
            standoff_position, facing_yaw, inside_position, centring_yaw, crumb_position, crumb_yaw, crumb_valid,
        )

    def _reference(
        self,
        robot_position_w: torch.Tensor,
        robot_yaw_w: torch.Tensor,
        gps_target_w: torch.Tensor,
        station_position: torch.Tensor,
        station_yaw: torch.Tensor,
        standoff_position: torch.Tensor,
        facing_yaw: torch.Tensor,
        inside_position: torch.Tensor,
        centring_yaw: torch.Tensor,
        crumb_position: torch.Tensor,
        crumb_yaw: torch.Tensor,
        crumb_valid: torch.Tensor,
    ) -> MissionReference:
        """Select one reference per environment from the phase."""

        position = robot_position_w.clone()
        yaw = robot_yaw_w.clone()

        def pick(phase: MissionPhase, target: torch.Tensor, heading: torch.Tensor) -> None:
            nonlocal position, yaw
            mask = self._in(phase)
            position = torch.where(mask[:, None], target, position)
            yaw = torch.where(mask, heading, yaw)

        bearing_in = torch.atan2(
            gps_target_w[:, 1] - robot_position_w[:, 1], gps_target_w[:, 0] - robot_position_w[:, 0]
        )
        # Aim just inside the region: a reference exactly on the boundary leaves the
        # aircraft hovering on the arrival test.
        entry_radius = self.cfg.region_radius_m * self.cfg.region_entry_fraction
        edge = gps_target_w.clone()
        edge[:, 0] = edge[:, 0] + entry_radius * torch.cos(self.ring_start_yaw)
        edge[:, 1] = edge[:, 1] + entry_radius * torch.sin(self.ring_start_yaw)
        edge[:, 2] = gps_target_w[:, 2] + self.cfg.floor_band_height_m * 0.5
        pick(MissionPhase.TRANSIT, edge, bearing_in)
        pick(MissionPhase.SURVEY, station_position, station_yaw)
        pick(MissionPhase.APPROACH, standoff_position, facing_yaw)
        pick(MissionPhase.VALIDATE, standoff_position, facing_yaw)
        pick(MissionPhase.ENTRY, inside_position, facing_yaw)
        pick(MissionPhase.OBSERVE, robot_position_w, centring_yaw)
        pick(MissionPhase.EXIT, torch.where(crumb_valid[:, None], crumb_position, robot_position_w), crumb_yaw)
        pick(MissionPhase.RESUME_SURVEY, self.checkpoint_w, station_yaw)
        pick(MissionPhase.NEXT_FLOOR, station_position, station_yaw)

        exterior = self._in(MissionPhase.TRANSIT) | self._in(MissionPhase.SURVEY) | self._in(MissionPhase.RESUME_SURVEY)
        return MissionReference(
            position_w=position,
            yaw_w=wrap_pi(yaw),
            phase=self.phase.clone(),
            entry_authorized=self._entry_authorized.clone(),
            target_valid=self.active_slot >= 0,
            detector_enabled=exterior | self._in(MissionPhase.APPROACH) | self._in(MissionPhase.OBSERVE),
            accepts_proposals=exterior,
            observation_progress=(self.person.held_s / self.person.cfg.hold_target_s).clamp(0.0, 1.0),
        )

    def observation(self, robot_position_w: torch.Tensor) -> torch.Tensor:
        """Fixed-width mission features: phase one-hot, ledger summary, hold, and route."""

        phase = torch.nn.functional.one_hot(self.phase, num_classes=PHASE_COUNT).float()
        return torch.cat(
            (
                phase,
                self.ledger.observation(robot_position_w),
                self.person.observation(),
                self.breadcrumbs.observation(robot_position_w),
                self._entry_authorized.float()[:, None],
            ),
            dim=-1,
        )
