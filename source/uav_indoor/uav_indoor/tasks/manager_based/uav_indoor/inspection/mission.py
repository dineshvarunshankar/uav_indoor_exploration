"""Vectorized deterministic mission supervisor for building-opening inspection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

from .ledger import LedgerCfg, OpeningLedger, OpeningStatus
from .observation import ObservationCfg, PrimaryPersonObservation
from .proposal_search import ProposalSearchRegistry, ProposalSearchStatus, project_exterior_station
from .recovery import BreadcrumbBuffer, RecoveryCfg


class MissionStage(IntEnum):
    """Curriculum stage. Each stage adds the next part of the mission."""

    TRANSIT_ONLY = 0
    SURVEY_ONLY = 1
    VALIDATE_ONLY = 2
    FULL = 3


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
    NEXT_FACADE = 9
    DONE = 10


PHASE_COUNT = len(MissionPhase)


# Mutable Hydra configuration.
@dataclass
class BuildingMissionCfg:
    """Mission geometry and timing."""

    region_radius_m: float = 8.0
    region_entry_fraction: float = 0.85
    # Facade sweep geometry.
    station_pitch_m: float = 6.0
    sweep_half_span_m: float = 30.0
    blank_stations_before_turn: int = 3
    standoff_m: float = 6.0  # facade clearance after the exterior standoff/survey point is reached
    # Approach standoff within ToF range.
    approach_standoff_m: float = 2.2
    station_tolerance_m: float = 1.0
    scan_dwell_s: float = 2.0
    # Search yaw sequence: 0 -> +half_span -> -half_span -> 0.
    proposal_search_yaw_half_span_rad: float | None = None
    proposal_search_yaw_rate_rad_s: float | None = None
    approach_tolerance_m: float = 0.8
    validation_hits: int = 3
    validation_timeout_s: float = 8.0
    corridor_clearance_m: float = 1.2
    # Minimum depth beyond the facade, in meters.
    corridor_depth_margin_m: float = 0.5
    # Mirrors ``TofCfg.reliable_range_m``. Missing returns remain unknown at any range.
    tof_reliable_range_m: float = 2.5
    # Maximum camera-axis error for entry, in radians.
    entry_aim_tolerance_rad: float = 0.15
    entry_depth_m: float = 0.9
    entry_tolerance_m: float = 0.35
    empty_observe_s: float = 10.0
    observe_timeout_s: float = 30.0
    exit_timeout_s: float = 20.0
    floor_band_height_m: float = 3.0
    max_floors: int = 3
    facades_per_floor: int = 4
    facade_fit_agreement: float = 0.7  # about 45 degrees
    facade_fit_max_shift_m: float = 2.0
    commit_radius_m: float = 10.0
    detector_activation_radius_m: float = 20.0
    horizontal_fov_rad: float = math.radians(79.0)
    # Initial curriculum stage.
    initial_stage: int = int(MissionStage.TRANSIT_ONLY)
    # Isolated-entry and curriculum controls.
    lock_stage: bool = False
    commit_without_dwell: bool = False
    hold_active_target: bool = False


@dataclass
class MissionReference:
    """World-frame reference and phase metadata emitted for local navigation."""

    position_w: torch.Tensor
    yaw_w: torch.Tensor
    phase: torch.Tensor
    entry_authorized: torch.Tensor
    crossing_opening: torch.Tensor
    target_valid: torch.Tensor
    detector_enabled: torch.Tensor
    accepts_proposals: torch.Tensor
    observation_progress: torch.Tensor


def wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap radians to [-pi, pi)."""

    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


class BuildingMissionSupervisor:
    """Batch mission state machine for local transit, facade survey, entry, and recovery."""

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
        search_yaw_values = (
            self.cfg.proposal_search_yaw_half_span_rad,
            self.cfg.proposal_search_yaw_rate_rad_s,
        )
        if (search_yaw_values[0] is None) != (search_yaw_values[1] is None):
            raise ValueError("proposal search yaw half-span and rate must be configured together")
        if any(
            value is not None and (not math.isfinite(value) or value <= 0.0)
            for value in search_yaw_values
        ):
            raise ValueError("proposal search yaw half-span and rate must be finite and positive")
        if not int(MissionStage.TRANSIT_ONLY) <= self.cfg.initial_stage <= int(MissionStage.FULL):
            raise ValueError(
                f"initial_stage must be between {int(MissionStage.TRANSIT_ONLY)} and "
                f"{int(MissionStage.FULL)}, got {self.cfg.initial_stage}"
            )

        self.ledger = OpeningLedger(num_envs, device, ledger_cfg)
        self.person = PrimaryPersonObservation(num_envs, device, observation_cfg)
        self.proposal_search = ProposalSearchRegistry(num_envs, device)
        self.breadcrumbs = BreadcrumbBuffer(num_envs, device, recovery_cfg)

        self.phase = torch.full((num_envs,), int(MissionPhase.TRANSIT), dtype=torch.long, device=self.device)
        self.phase_time = torch.zeros(num_envs, device=self.device)
        self.at_station = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.floor = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.facade_index = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Facade-turn direction and last corner.
        self.facade_turn_sign = torch.ones(num_envs, device=self.device)
        self.corner_w = torch.zeros(num_envs, 3, device=self.device)
        # Signed station offset along the facade.
        self.station_offset = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.blank_streak = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Completed station count.
        self.stations_scanned = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.facade_normal_w = torch.zeros(num_envs, 3, device=self.device)
        self.facade_normal_w[:, 0] = -1.0
        self.floor_start_normal_w = self.facade_normal_w.clone()
        self.sweep_anchor_w = torch.zeros(num_envs, 3, device=self.device)
        # Initial facade offset for bounded refinement.
        self.facade_offset_seed_m = torch.zeros(num_envs, device=self.device)
        # Per-environment station standoff.
        self.station_standoff_m = torch.full((num_envs,), self.cfg.standoff_m, device=self.device)
        self.facade_known = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.station_dwell_s = torch.zeros(num_envs, device=self.device)
        self.ring_start_yaw = torch.zeros(num_envs, device=self.device)
        self.survey_direction = torch.ones(num_envs, device=self.device)
        self.building_center_w = torch.zeros(num_envs, 3, device=self.device)
        self.checkpoint_w = torch.zeros(num_envs, 3, device=self.device)
        self.observation_position_w = torch.zeros(num_envs, 3, device=self.device)
        self.active_slot = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.active_refreshed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.validation_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.validation_target_slot = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.validation_samples_total = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.validation_tof_invalid_total = torch.zeros_like(self.validation_samples_total)
        self.validation_tof_short_total = torch.zeros_like(self.validation_samples_total)
        self.validation_clearance_low_total = torch.zeros_like(self.validation_samples_total)
        self.validation_aim_off_total = torch.zeros_like(self.validation_samples_total)
        self.validation_clearance_unknown_total = torch.zeros_like(self.validation_samples_total)
        self.validation_accepted_total = torch.zeros_like(self.validation_samples_total)
        self.validation_success_total = torch.zeros_like(self.validation_samples_total)
        # Validation timeout count.
        self.validation_rejected_total = torch.zeros_like(self.validation_samples_total)
        self.recovery_failed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._entry_authorized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        # External facade-orientation flag.
        self.facade_hint_known = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.observation_by_hold = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.observation_by_empty = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.observation_by_timeout = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.observed_hold_s = torch.zeros(num_envs, device=self.device)
        self._crossing_opening = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.search_hint_slot = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.proposal_search_elapsed_s = torch.zeros(num_envs, device=self.device)
        self.mission_success = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.mission_incomplete = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        # Initialize the curriculum.
        self.stage = torch.full((num_envs,), self.cfg.initial_stage, dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None, survey_direction: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.phase[env_ids] = int(MissionPhase.TRANSIT)
        self.phase_time[env_ids] = 0.0
        self.at_station[env_ids] = False
        self.floor[env_ids] = 0
        self.facade_index[env_ids] = 0
        self.facade_turn_sign[env_ids] = 1.0
        self.corner_w[env_ids] = 0.0
        self.station_offset[env_ids] = 0
        self.blank_streak[env_ids] = 0
        self.stations_scanned[env_ids] = 0
        self.facade_normal_w[env_ids] = 0.0
        self.facade_normal_w[env_ids, 0] = -1.0
        self.floor_start_normal_w[env_ids] = self.facade_normal_w[env_ids]
        self.sweep_anchor_w[env_ids] = 0.0
        self.facade_offset_seed_m[env_ids] = 0.0
        self.station_standoff_m[env_ids] = self.cfg.standoff_m
        self.observation_by_hold[env_ids] = False
        self.observation_by_empty[env_ids] = False
        self.observation_by_timeout[env_ids] = False
        self.observed_hold_s[env_ids] = 0.0
        self.facade_known[env_ids] = False
        self.station_dwell_s[env_ids] = 0.0
        self.ring_start_yaw[env_ids] = 0.0
        self.building_center_w[env_ids] = 0.0
        self.checkpoint_w[env_ids] = 0.0
        self.observation_position_w[env_ids] = 0.0
        self.active_slot[env_ids] = -1
        self.validation_count[env_ids] = 0
        self.validation_target_slot[env_ids] = -1
        self.validation_samples_total[env_ids] = 0
        self.validation_tof_invalid_total[env_ids] = 0
        self.validation_tof_short_total[env_ids] = 0
        self.validation_clearance_low_total[env_ids] = 0
        self.validation_aim_off_total[env_ids] = 0
        self.validation_clearance_unknown_total[env_ids] = 0
        self.validation_accepted_total[env_ids] = 0
        self.validation_success_total[env_ids] = 0
        self.validation_rejected_total[env_ids] = 0
        self.recovery_failed[env_ids] = False
        self._entry_authorized[env_ids] = False
        self._crossing_opening[env_ids] = False
        self.search_hint_slot[env_ids] = -1
        self.proposal_search_elapsed_s[env_ids] = 0.0
        self.mission_success[env_ids] = False
        self.mission_incomplete[env_ids] = False
        if survey_direction is None:
            survey_direction = torch.where(
                torch.rand(env_ids.shape[0], device=self.device) < 0.5,
                torch.full((env_ids.shape[0],), -1.0, device=self.device),
                torch.ones(env_ids.shape[0], device=self.device),
            )
        self.survey_direction[env_ids] = survey_direction.to(self.device)
        self.ledger.reset(env_ids)
        self.person.reset(env_ids)
        self.proposal_search.reset(env_ids)
        self.breadcrumbs.reset(env_ids)

    def set_stage(self, stage: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Set the curriculum stage for all or selected environments."""

        if env_ids is None:
            self.stage = stage.to(device=self.device, dtype=torch.long)
        else:
            self.stage[env_ids.to(self.device)] = stage.to(device=self.device, dtype=torch.long)

    def begin_at_survey(
        self,
        env_ids: torch.Tensor,
        anchor_w: torch.Tensor,
        facade_normal_w: torch.Tensor,
        station_standoff_m: torch.Tensor | None = None,
    ) -> None:
        """Start at a survey station; anchor_w lies on the wall and station_standoff_m gives the exterior distance."""

        ids = env_ids.to(self.device)
        anchor = anchor_w.to(self.device)
        normal = facade_normal_w.to(self.device)
        if station_standoff_m is not None:
            self.station_standoff_m[ids] = station_standoff_m.to(self.device)
        self.facade_normal_w[ids] = normal
        self.floor_start_normal_w[ids] = normal
        self.sweep_anchor_w[ids] = anchor
        self.building_center_w[ids] = anchor
        self.facade_offset_seed_m[ids] = (normal * anchor).sum(dim=-1)
        self.facade_known[ids] = True
        self.ring_start_yaw[ids] = torch.atan2(normal[:, 1], normal[:, 0])
        self.facade_index[ids] = 0
        self.station_offset[ids] = 0
        self.station_dwell_s[ids] = 0.0
        self.phase[ids] = int(MissionPhase.SURVEY)
        self.phase_time[ids] = 0.0

    def _transition(self, mask: torch.Tensor, phase: MissionPhase) -> None:
        self.phase = torch.where(mask, torch.full_like(self.phase, int(phase)), self.phase)
        self.phase_time = torch.where(mask, torch.zeros_like(self.phase_time), self.phase_time)

    def _in(self, phase: MissionPhase) -> torch.Tensor:
        return self.phase == int(phase)

    @property
    def floor_height_w(self) -> torch.Tensor:
        """Height of the current floor band from the supplied first-band altitude."""

        return self.building_center_w[:, 2] + self.cfg.floor_band_height_m * self.floor.float()

    def station_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Position and facade-facing yaw of the current scan station."""

        tangent = torch.stack(
            (-self.facade_normal_w[:, 1], self.facade_normal_w[:, 0], torch.zeros_like(self.facade_normal_w[:, 0])),
            dim=-1,
        )
        along = self.station_offset.float() * self.cfg.station_pitch_m
        position = (
            self.sweep_anchor_w
            + tangent * along[:, None]
            + self.facade_normal_w * self.station_standoff_m[:, None]
        )
        position[:, 2] = self.floor_height_w
        facing = wrap_pi(torch.atan2(-self.facade_normal_w[:, 1], -self.facade_normal_w[:, 0]))
        return position, facing

    def _search_station_position(
        self,
        station_position: torch.Tensor,
        search_active: torch.Tensor,
    ) -> torch.Tensor:
        """Replace a sweep station with the active exterior proposal-search station."""

        if not self.proposal_search.proposal_count:
            return station_position
        selected_position = self.proposal_search.position_local.gather(
            1,
            self.search_hint_slot.clamp_min(0)[:, None, None].expand(-1, 1, 3),
        ).squeeze(1)
        search_station = project_exterior_station(
            selected_position,
            self.sweep_anchor_w,
            self.facade_normal_w,
            self.station_standoff_m,
            self.floor_height_w,
        )
        return torch.where(search_active[:, None], search_station, station_position)

    def proposal_search_yaw(self, facade_yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the deterministic proposal-search heading and completion state."""

        half_span = self.cfg.proposal_search_yaw_half_span_rad
        rate = self.cfg.proposal_search_yaw_rate_rad_s
        if half_span is None or rate is None:
            return facade_yaw, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        travel = self.proposal_search_elapsed_s * rate
        offset = torch.where(
            travel <= half_span,
            travel,
            torch.where(
                travel <= 3.0 * half_span,
                2.0 * half_span - travel,
                travel - 4.0 * half_span,
            ),
        )
        complete = travel >= 4.0 * half_span
        offset = torch.where(complete, torch.zeros_like(offset), offset)
        return wrap_pi(facade_yaw + offset), complete

    @property
    def facade_offset_m(self) -> torch.Tensor:
        """Signed distance of the facade plane from the origin, along its normal."""

        return (self.facade_normal_w * self.sweep_anchor_w).sum(dim=-1)

    def _project_to_facade(
        self, origin_w: torch.Tensor, ray_w: torch.Tensor, max_range_m: float = 80.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Where each bearing meets the facade, and whether that meeting is usable."""

        normal = self.facade_normal_w[:, None, :]
        denominator = (ray_w * normal).sum(dim=-1)
        numerator = self.facade_offset_m[:, None] - (origin_w[:, None, :] * normal).sum(dim=-1)
        parallel = denominator.abs() < 1e-6
        distance = numerator / torch.where(parallel, torch.ones_like(denominator), denominator)
        usable = ~parallel & (distance > 0.0) & (distance < max_range_m) & self.facade_known[:, None]
        return origin_w[:, None, :] + ray_w * distance.unsqueeze(-1), usable

    def _record_facade_geometry(self, origin_w: torch.Tensor, angular_wh: torch.Tensor) -> None:
        """Give each solved entry the facade's normal and its metric size."""

        facing = wrap_pi(torch.atan2(-self.facade_normal_w[:, 1], -self.facade_normal_w[:, 0]))
        occupied = self.ledger.occupied
        self.ledger.approach_yaw = torch.where(
            occupied, facing[:, None].expand_as(self.ledger.approach_yaw), self.ledger.approach_yaw
        )

        range_m = torch.linalg.vector_norm(self.ledger.position_w - origin_w[:, None, :], dim=-1)
        seen = self.ledger.solved & occupied
        # Lower-bound opening size: angular extent times range.
        metric = 2.0 * range_m[..., None] * torch.tan(0.5 * angular_wh.mean(dim=1, keepdim=True))
        self.ledger.size_wh = torch.where(seen[..., None], metric.expand_as(self.ledger.size_wh), self.ledger.size_wh)

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
        building_standoff_w: torch.Tensor,
        opening_origin_w: torch.Tensor,
        opening_ray_w: torch.Tensor,
        opening_angular_wh: torch.Tensor,
        opening_track_id: torch.Tensor,
        opening_box_xywhn: torch.Tensor,
        opening_confidence: torch.Tensor,
        opening_valid: torch.Tensor,
        opening_box_centered: torch.Tensor,
        person_track_id: torch.Tensor,
        person_box_xywhn: torch.Tensor,
        person_confidence: torch.Tensor,
        person_age_s: torch.Tensor,
        person_valid: torch.Tensor,
        tof_valid: torch.Tensor,
        tof_clear: torch.Tensor,
        clearance_m: torch.Tensor,
        clearance_known: torch.Tensor,
        avoidance_active: torch.Tensor,
        time_s: float,
        maximum_association_residual_m: float | None = None,
        # Detection validity for active-target refresh; defaults to opening_valid.
        opening_detected: torch.Tensor | None = None,
        # Optional facade orientation hint; offset remains adjustable.
        facade_normal_hint: torch.Tensor | None = None,
        facade_normal_w: torch.Tensor | None = None,
        facade_offset_m: torch.Tensor | None = None,
        facade_valid: torch.Tensor | None = None,
    ) -> MissionReference:
        """Advance mission state and return the current local-motion reference."""

        self.phase_time = self.phase_time + dt

        # TRANSIT: supplied standoff target.
        transit = self._in(MissionPhase.TRANSIT)
        approach_side = torch.atan2(
            robot_position_w[:, 1] - building_standoff_w[:, 1],
            robot_position_w[:, 0] - building_standoff_w[:, 0],
        )
        transit_standoff_w = building_standoff_w.clone()
        at_standoff = (
            torch.linalg.vector_norm(robot_position_w - transit_standoff_w, dim=-1) <= self.cfg.station_tolerance_m
        )
        arrived = transit & at_standoff

        # Seed facade orientation from the approach bearing.
        seed_side = torch.where(arrived, self.ring_start_yaw, approach_side)
        seed_normal = torch.stack((torch.cos(seed_side), torch.sin(seed_side), torch.zeros_like(seed_side)), dim=-1)
        hint_supplied = facade_normal_hint is not None
        if hint_supplied:
            seed_normal = facade_normal_hint / facade_normal_hint.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        self.facade_hint_known = torch.full_like(self.facade_known, hint_supplied)
        self.facade_normal_w = torch.where(arrived[:, None], seed_normal, self.facade_normal_w)
        self.floor_start_normal_w = torch.where(arrived[:, None], seed_normal, self.floor_start_normal_w)
        survey_anchor_w = building_standoff_w - seed_normal * self.cfg.standoff_m
        # Store the first-facade anchor.
        self.building_center_w = torch.where(arrived[:, None], survey_anchor_w, self.building_center_w)
        self.sweep_anchor_w = torch.where(arrived[:, None], survey_anchor_w, self.sweep_anchor_w)
        self.facade_offset_seed_m = torch.where(
            arrived, (seed_normal * survey_anchor_w).sum(dim=-1), self.facade_offset_seed_m
        )
        self.facade_known = self.facade_known | arrived
        facade_adopted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if facade_valid is not None:
            # Accept fits for the current facade.
            agrees = (facade_normal_w * self.facade_normal_w).sum(dim=-1) > self.cfg.facade_fit_agreement
            # Bound offset refinement against the initial seed.
            limit = self.cfg.facade_fit_max_shift_m
            bounded_offset = torch.minimum(
                torch.maximum(facade_offset_m, self.facade_offset_seed_m - limit),
                self.facade_offset_seed_m + limit,
            )
            shift = (bounded_offset - self.facade_offset_m).abs()
            adopt = facade_valid & self.facade_known & agrees & (shift <= limit)
            facade_adopted = adopt
            # Preserve supplied facade orientation.
            rotate = adopt & ~self.facade_hint_known
            self.facade_normal_w = torch.where(rotate[:, None], facade_normal_w, self.facade_normal_w)
            # Keep the anchor on the wall: project it onto the fitted plane.
            gap = (self.sweep_anchor_w * self.facade_normal_w).sum(dim=-1) - bounded_offset
            projected = self.sweep_anchor_w - self.facade_normal_w * gap[:, None]
            self.sweep_anchor_w = torch.where(adopt[:, None], projected, self.sweep_anchor_w)
        # Retain the approach bearing until arrival.
        self.ring_start_yaw = torch.where(transit & ~arrived, approach_side, self.ring_start_yaw)
        self.facade_index = torch.where(arrived, torch.zeros_like(self.facade_index), self.facade_index)
        self.facade_turn_sign = torch.where(arrived, self.survey_direction, self.facade_turn_sign)
        self.station_offset = torch.where(arrived, torch.zeros_like(self.station_offset), self.station_offset)
        self.station_dwell_s = torch.where(arrived, torch.zeros_like(self.station_dwell_s), self.station_dwell_s)
        transit_only_done = arrived & (self.stage == int(MissionStage.TRANSIT_ONLY))
        self._transition(transit_only_done, MissionPhase.DONE)
        self._transition(arrived & ~transit_only_done, MissionPhase.SURVEY)

        # SURVEY: stop, scan the facade, then step along the wall.
        occupied_before = self.ledger.occupied.sum(dim=1)
        survey = self._in(MissionPhase.SURVEY)
        station_position, station_yaw = self.station_pose()
        selected_hint, has_hint = self.proposal_search.select_pending(
            station_position,
            self.floor_height_w,
            self.cfg.floor_band_height_m,
        )
        start_hint_search = survey & (self.search_hint_slot < 0) & has_hint
        self.search_hint_slot = torch.where(start_hint_search, selected_hint, self.search_hint_slot)
        self.proposal_search.mark(
            self.search_hint_slot.clamp_min(0),
            ProposalSearchStatus.SEARCHING,
            start_hint_search,
        )
        search_active = survey & (self.search_hint_slot >= 0)
        station_position = self._search_station_position(station_position, search_active)
        at_station = (
            torch.linalg.vector_norm(robot_position_w - station_position, dim=-1) <= self.cfg.station_tolerance_m
        )
        scanning = survey & at_station
        # Station-arrival telemetry.
        self.at_station = at_station
        self.station_dwell_s = torch.where(scanning, self.station_dwell_s + dt, self.station_dwell_s)
        self.proposal_search_elapsed_s = torch.where(
            scanning & search_active,
            self.proposal_search_elapsed_s + dt,
            self.proposal_search_elapsed_s,
        )
        seen_w, seen_valid = self._project_to_facade(opening_origin_w, opening_ray_w)
        self.ledger.observe_bearings(
            origin_w=opening_origin_w,
            ray_w=opening_ray_w,
            seen_w=seen_w,
            seen_valid=seen_valid & opening_valid & scanning[:, None],
            confidence=opening_confidence,
            floor=self.floor[:, None].expand_as(opening_confidence),
            time_s=time_s,
        )
        # Refresh active-target identity.
        holding_target = (
            self._in(MissionPhase.APPROACH) | self._in(MissionPhase.VALIDATE) | self._in(MissionPhase.ENTRY)
        )
        # Use detection validity outside SURVEY.
        detected = opening_valid if opening_detected is None else opening_detected
        self.active_refreshed = self.ledger.refresh_active(
            self.active_slot,
            holding_target & (self.active_slot >= 0),
            opening_track_id,
            opening_box_xywhn,
            seen_w,
            seen_valid & detected,
            time_s,
        )

        approach_yaw_tracks = wrap_pi(
            torch.atan2(-self.facade_normal_w[:, 1], -self.facade_normal_w[:, 0])
        )[:, None].expand_as(opening_confidence)
        self.ledger.observe_tracks(
            position_w=seen_w,
            approach_yaw=approach_yaw_tracks,
            size_wh=opening_angular_wh,
            floor=self.floor[:, None].expand_as(opening_confidence),
            track_id=opening_track_id,
            box_xywhn=opening_box_xywhn,
            confidence=opening_confidence,
            valid=seen_valid & opening_valid & scanning[:, None],
            time_s=time_s,
        )
        # Prune solved candidates against the accepted facade fit.
        self.ledger.prune_off_facade(self.facade_normal_w, self.facade_offset_m, facade_adopted)
        self._record_facade_geometry(opening_origin_w, opening_angular_wh)

        ledger_slots = torch.arange(
            self.ledger.max_openings,
            device=self.device,
            dtype=torch.long,
        )[None, :].expand(self.num_envs, -1)
        detector_backed = (self.ledger.source == 0) & self.ledger.approachable
        active_hint = self.proposal_search.valid & (
            torch.arange(self.proposal_search.proposal_count, device=self.device)[None, :]
            == self.search_hint_slot[:, None]
        ) & search_active[:, None]
        self.proposal_search.associate(
            self.ledger.position_w,
            ledger_slots,
            detector_backed,
            maximum_association_residual_m,
            searchable_hint=active_hint,
        )
        self.proposal_search.sync_terminal(
            self.ledger.status,
            int(OpeningStatus.INSPECTED),
            int(OpeningStatus.UNSAFE),
        )

        dwelled = self.station_dwell_s >= self.cfg.scan_dwell_s
        scanned = scanning & (dwelled | self.cfg.commit_without_dwell)
        ordinary_selected, ordinary_candidate = self.ledger.select_candidate(
            station_position,
            self.facade_normal_w,
            self.survey_direction,
            self.floor,
            facade_offset_m=self.facade_offset_m,
        )
        matched_selection = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        hint_matched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.proposal_search.proposal_count:
            hint_index = self.search_hint_slot.clamp_min(0)[:, None]
            matched_selection = self.proposal_search.matched_slot.gather(1, hint_index).squeeze(1)
            hint_status = self.proposal_search.status.gather(1, hint_index).squeeze(1)
            hint_matched = (
                search_active
                & (hint_status == int(ProposalSearchStatus.MATCHED))
                & (matched_selection >= 0)
            )
        _, sweep_complete = self.proposal_search_yaw(station_yaw)
        search_finished_unmatched = search_active & ~hint_matched & sweep_complete
        self.proposal_search.mark(
            self.search_hint_slot.clamp_min(0),
            ProposalSearchStatus.UNRESOLVED,
            search_finished_unmatched,
        )
        selected = torch.where(search_active, matched_selection.clamp_min(0), ordinary_selected)
        has_candidate = torch.where(search_active, hint_matched, ordinary_candidate)
        commit = scanned & has_candidate
        surveyed = commit & (self.stage == int(MissionStage.SURVEY_ONLY))
        approach_commit = commit & ~surveyed
        if self.cfg.hold_active_target:
            approach_commit = approach_commit & (self.active_slot < 0)
        self.active_slot = torch.where(approach_commit, selected, self.active_slot)
        self.checkpoint_w = torch.where(approach_commit[:, None], station_position, self.checkpoint_w)
        self.ledger.mark(selected, OpeningStatus.ACTIVE, approach_commit)
        self._transition(surveyed, MissionPhase.DONE)
        self._transition(approach_commit, MissionPhase.APPROACH)

        finished_hint_search = (commit | search_finished_unmatched) & search_active
        self.search_hint_slot = torch.where(
            finished_hint_search,
            torch.full_like(self.search_hint_slot, -1),
            self.search_hint_slot,
        )
        self.proposal_search_elapsed_s = torch.where(
            finished_hint_search,
            torch.zeros_like(self.proposal_search_elapsed_s),
            self.proposal_search_elapsed_s,
        )
        self.station_dwell_s = torch.where(
            finished_hint_search,
            torch.zeros_like(self.station_dwell_s),
            self.station_dwell_s,
        )

        # Advance the survey station.
        advance_station = scanning & dwelled & ~has_candidate & ~search_active
        found_something = self.ledger.occupied.sum(dim=1) > occupied_before
        self.blank_streak = torch.where(advance_station & ~found_something, self.blank_streak + 1, self.blank_streak)
        self.blank_streak = torch.where(found_something, torch.zeros_like(self.blank_streak), self.blank_streak)
        self.station_offset = self.station_offset + (advance_station.long() * self.survey_direction.long())
        self.stations_scanned = self.stations_scanned + advance_station.long()
        self.station_dwell_s = torch.where(
            advance_station, torch.zeros_like(self.station_dwell_s), self.station_dwell_s
        )

        # Advance the facade or floor at the sweep limit.
        reached_span = self.station_offset.abs().float() * self.cfg.station_pitch_m >= self.cfg.sweep_half_span_m
        run_done = (
            self._in(MissionPhase.SURVEY)
            & advance_station
            & (reached_span | (self.blank_streak >= self.cfg.blank_stations_before_turn))
        )
        last_facade = self.facade_index + 1 >= self.cfg.facades_per_floor
        # Use the turn point as the next facade anchor.
        self.corner_w = torch.where(run_done[:, None], robot_position_w, self.corner_w)
        self.survey_direction = torch.where(run_done, -self.survey_direction, self.survey_direction)
        self.blank_streak = torch.where(run_done, torch.zeros_like(self.blank_streak), self.blank_streak)
        self._transition(run_done & ~last_facade, MissionPhase.NEXT_FACADE)
        self._transition(run_done & last_facade, MissionPhase.NEXT_FLOOR)

        # APPROACH and VALIDATE route targets.
        opening_w = self._active(self.ledger.position_w)
        approach_yaw = self._active(self.ledger.approach_yaw)
        # Approach standoff and yaw.
        standoff_position = opening_w.clone()
        standoff_position[:, 0] = standoff_position[:, 0] - self.cfg.approach_standoff_m * torch.cos(approach_yaw)
        standoff_position[:, 1] = standoff_position[:, 1] - self.cfg.approach_standoff_m * torch.sin(approach_yaw)
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

        # Require consecutive valid ToF samples for entry.
        validate = self._in(MissionPhase.VALIDATE)
        # Entry aim uses the selected live RGB box.
        aimed = opening_box_centered
        blocked = tof_valid & ~tof_clear
        # Use opening-specific ToF evidence.
        evidence = tof_valid & tof_clear & aimed

        self.validation_samples_total += validate.long()
        self.validation_tof_invalid_total += (validate & ~tof_valid).long()
        self.validation_tof_short_total += (validate & blocked).long()
        measured = validate & tof_valid & tof_clear
        self.validation_aim_off_total += (measured & ~aimed).long()
        self.validation_accepted_total += (validate & evidence).long()
        target_changed = validate & (self.validation_target_slot != self.active_slot)
        prior_count = torch.where(
            target_changed, torch.zeros_like(self.validation_count), self.validation_count
        )
        cleared = torch.where(validate, torch.zeros_like(prior_count), prior_count)
        self.validation_count = torch.where(validate & evidence, prior_count + 1, cleared)
        self.validation_target_slot = torch.where(
            validate, self.active_slot, torch.full_like(self.validation_target_slot, -1)
        )
        authorized = validate & (self.validation_count >= self.cfg.validation_hits)
        self.validation_success_total += authorized.long()
        self.ledger.mark_validated(self.active_slot.clamp_min(0), authorized)
        validated_only = authorized & (self.stage == int(MissionStage.VALIDATE_ONLY))
        self._transition(validated_only, MissionPhase.DONE)
        authorized = authorized & ~validated_only
        self.breadcrumbs.start(authorized)
        self._transition(authorized, MissionPhase.ENTRY)

        rejected = validate & ~authorized & (self.phase_time >= self.cfg.validation_timeout_s)
        self.validation_rejected_total += rejected.long()
        self.ledger.mark(self.active_slot.clamp_min(0), OpeningStatus.UNSAFE, rejected)
        self._transition(rejected, MissionPhase.RESUME_SURVEY)

        # ENTRY: cross the authorized corridor and record the protected route.
        entry = self._in(MissionPhase.ENTRY)
        inside_position = opening_w.clone()
        inside_position[:, 0] = inside_position[:, 0] + self.cfg.entry_depth_m * torch.cos(approach_yaw)
        inside_position[:, 1] = inside_position[:, 1] + self.cfg.entry_depth_m * torch.sin(approach_yaw)
        self.breadcrumbs.push(robot_position_w, robot_yaw_w, clearance_m, time_s, entry)
        inside = entry & (
            torch.linalg.vector_norm(robot_position_w - inside_position, dim=-1) <= self.cfg.entry_tolerance_m
        )
        self.breadcrumbs.seal(inside)
        self.observation_position_w = torch.where(
            inside[:, None], robot_position_w, self.observation_position_w
        )
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
        empty_look_done = (self.person.primary_id < 0) & (self.phase_time >= self.cfg.empty_observe_s)
        observation_timed_out = observe & (self.phase_time >= self.cfg.observe_timeout_s)
        observation_done = observe & (self.person.complete | empty_look_done | observation_timed_out)
        # Record observation completion mode.
        self.observation_by_hold |= observe & self.person.complete
        self.observation_by_empty |= observe & empty_look_done
        observation_failed = observation_timed_out & ~(self.person.complete | empty_look_done)
        self.observation_by_timeout |= observation_failed
        # Mark incomplete observations UNSAFE before exit.
        self.ledger.mark(self.active_slot.clamp_min(0), OpeningStatus.UNSAFE, observation_failed)
        self.observed_hold_s = torch.maximum(self.observed_hold_s, self.person.held_s)
        self._transition(observation_done, MissionPhase.EXIT)

        # EXIT: fly the saved corridor in reverse while the forward sensors face travel.
        exit_phase = self._in(MissionPhase.EXIT)
        crumb_position, crumb_yaw, crumb_valid = self.breadcrumbs.next_crumb()
        exit_delta = crumb_position[:, :2] - robot_position_w[:, :2]
        exit_distance = torch.linalg.vector_norm(exit_delta, dim=-1)
        exit_travel_yaw = torch.atan2(exit_delta[:, 1], exit_delta[:, 0])
        # Use reversed entry yaw at zero displacement.
        exit_yaw = torch.where(
            exit_distance > 1.0e-3, exit_travel_yaw, wrap_pi(crumb_yaw + torch.pi)
        )
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

        # RESUME_SURVEY: return to the exterior checkpoint.
        resume = self._in(MissionPhase.RESUME_SURVEY)
        back = resume & (
            torch.linalg.vector_norm(robot_position_w - self.checkpoint_w, dim=-1) <= self.cfg.station_tolerance_m
        )
        self.active_slot = torch.where(back, torch.full_like(self.active_slot, -1), self.active_slot)
        self.station_dwell_s = torch.where(back, torch.zeros_like(self.station_dwell_s), self.station_dwell_s)
        self.person.clear(back)
        self._transition(back, MissionPhase.SURVEY)

        # NEXT_FACADE: rotate around the inferred building anchor to the next exterior side.
        next_facade = self._in(MissionPhase.NEXT_FACADE)
        # Use a fixed facade-turn direction.
        sign = self.facade_turn_sign
        n = self.facade_normal_w
        rotated = torch.stack((-sign * n[:, 1], sign * n[:, 0], torch.zeros_like(n[:, 0])), dim=-1)
        self.facade_normal_w = torch.where(next_facade[:, None], rotated, self.facade_normal_w)
        self.sweep_anchor_w = torch.where(next_facade[:, None], self.corner_w, self.sweep_anchor_w)
        self.facade_offset_seed_m = torch.where(
            next_facade, (self.facade_normal_w * self.corner_w).sum(dim=-1), self.facade_offset_seed_m
        )
        self.facade_index = self.facade_index + next_facade.long()
        self.station_offset = torch.where(next_facade, torch.zeros_like(self.station_offset), self.station_offset)
        self.station_dwell_s = torch.where(next_facade, torch.zeros_like(self.station_dwell_s), self.station_dwell_s)
        self._transition(next_facade, MissionPhase.SURVEY)

        # NEXT_FLOOR: start the facade set one band higher, or finish the building.
        next_floor = self._in(MissionPhase.NEXT_FLOOR)
        self.floor = self.floor + next_floor.long()
        self.facade_index = torch.where(next_floor, torch.zeros_like(self.facade_index), self.facade_index)
        self.facade_normal_w = torch.where(next_floor[:, None], self.floor_start_normal_w, self.facade_normal_w)
        self.sweep_anchor_w = torch.where(next_floor[:, None], self.building_center_w, self.sweep_anchor_w)
        self.facade_offset_seed_m = torch.where(
            next_floor,
            (self.facade_normal_w * self.building_center_w).sum(dim=-1),
            self.facade_offset_seed_m,
        )
        self.station_offset = torch.where(next_floor, torch.zeros_like(self.station_offset), self.station_offset)
        self.station_dwell_s = torch.where(next_floor, torch.zeros_like(self.station_dwell_s), self.station_dwell_s)
        finished = next_floor & (self.floor >= self.cfg.max_floors)
        full_mission_finished = finished & (self.stage == int(MissionStage.FULL))
        self.proposal_search.mark_remaining_unresolved(full_mission_finished)
        successful = self.ledger.building_complete() & self.proposal_search.success_ready & ~self.recovery_failed
        self.mission_success = torch.where(full_mission_finished, successful, self.mission_success)
        self.mission_incomplete = torch.where(full_mission_finished, ~successful, self.mission_incomplete)
        self._transition(next_floor & ~finished, MissionPhase.SURVEY)
        self._transition(finished, MissionPhase.DONE)

        station_position, station_yaw = self.station_pose()
        reference_search_active = self._in(MissionPhase.SURVEY) & (self.search_hint_slot >= 0)
        station_position = self._search_station_position(station_position, reference_search_active)
        search_yaw, _ = self.proposal_search_yaw(station_yaw)
        station_yaw = torch.where(reference_search_active, search_yaw, station_yaw)
        return self._reference(
            robot_position_w,
            robot_yaw_w,
            building_standoff_w,
            station_position,
            station_yaw,
            standoff_position,
            facing_yaw,
            inside_position,
            centring_yaw,
            crumb_position,
            exit_yaw,
            crumb_valid,
        )

    def _reference(
        self,
        robot_position_w: torch.Tensor,
        robot_yaw_w: torch.Tensor,
        building_standoff_w: torch.Tensor,
        station_position: torch.Tensor,
        station_yaw: torch.Tensor,
        standoff_position: torch.Tensor,
        facing_yaw: torch.Tensor,
        inside_position: torch.Tensor,
        centring_yaw: torch.Tensor,
        crumb_position: torch.Tensor,
        exit_yaw: torch.Tensor,
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
            building_standoff_w[:, 1] - robot_position_w[:, 1],
            building_standoff_w[:, 0] - robot_position_w[:, 0],
        )
        standoff = building_standoff_w.clone()
        pick(MissionPhase.TRANSIT, standoff, bearing_in)
        pick(MissionPhase.SURVEY, station_position, station_yaw)
        pick(MissionPhase.APPROACH, standoff_position, facing_yaw)
        pick(MissionPhase.VALIDATE, standoff_position, facing_yaw)
        pick(MissionPhase.ENTRY, inside_position, facing_yaw)
        pick(MissionPhase.OBSERVE, self.observation_position_w, centring_yaw)
        pick(MissionPhase.EXIT, torch.where(crumb_valid[:, None], crumb_position, robot_position_w), exit_yaw)
        pick(MissionPhase.RESUME_SURVEY, self.checkpoint_w, station_yaw)
        pick(MissionPhase.NEXT_FLOOR, station_position, station_yaw)
        pick(MissionPhase.NEXT_FACADE, station_position, station_yaw)

        # Authorization and crossing flags use the updated phase.
        self._entry_authorized = (
            self._in(MissionPhase.ENTRY) | self._in(MissionPhase.OBSERVE) | self._in(MissionPhase.EXIT)
        )
        self._crossing_opening = self._in(MissionPhase.ENTRY) | self._in(MissionPhase.EXIT)

        transit_near_target = self._in(MissionPhase.TRANSIT) & (
            torch.linalg.vector_norm(robot_position_w - building_standoff_w, dim=-1)
            <= self.cfg.detector_activation_radius_m
        )
        survey = self._in(MissionPhase.SURVEY)
        return MissionReference(
            position_w=position,
            yaw_w=wrap_pi(yaw),
            phase=self.phase.clone(),
            entry_authorized=self._entry_authorized.clone(),
            crossing_opening=self._crossing_opening.clone(),
            target_valid=self.active_slot >= 0,
            detector_enabled=(
                transit_near_target
                | survey
                | self._in(MissionPhase.APPROACH)
                | self._in(MissionPhase.VALIDATE)
                | self._in(MissionPhase.OBSERVE)
            ),
            # Admit ledger observations only during stationary SURVEY.
            accepts_proposals=survey,
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
