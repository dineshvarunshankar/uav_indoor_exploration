"""Vectorized deterministic mission supervisor for casualty inspection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch


class MissionPhase(IntEnum):
    """Mission phases owned by deterministic logic."""

    APPROACH = 0
    OBSERVE = 1
    SEARCH_FACE = 2
    STAGE = 3
    RECORD = 4
    COMPLETE = 5
    DONE = 6


STAGE_TERMINAL_PHASE = (6, 1, 2, 3, 4, 4, 4, 4)
STAGE_TARGET_COUNT = (1, 1, 1, 1, 1, 1, 2, 10)
STAGE_HAZARDS_ACTIVE = (False, False, False, False, False, True, True, True)
FINAL_STAGE = len(STAGE_TARGET_COUNT) - 1


@dataclass(frozen=True)
class MissionSupervisorCfg:
    """Timing and geometric constraints for the deterministic supervisor."""

    gps_region_radius_m: float = 3.0
    observation_s: float = 1.0
    search_timeout_s: float = 12.0
    search_orbit_rate_rad_s: float = 0.35
    standoff_m: float = 2.25
    vertical_fov_deg: float = 66.0
    min_inspection_height_m: float = 0.8
    arc_half_angle_deg: float = 15.0
    arc_duration_s: float = 10.0
    stage_position_tolerance_m: float = 0.35
    stage_yaw_tolerance_rad: float = 0.20
    record_position_tolerance_m: float = 0.55
    max_tracking_age_s: float = 0.5
    max_retries: int = 3


@dataclass
class MissionReference:
    """World-frame reference and phase metadata emitted for local navigation."""

    position_w: torch.Tensor
    yaw_w: torch.Tensor
    phase: torch.Tensor
    target_valid: torch.Tensor
    recording: torch.Tensor
    arc_progress: torch.Tensor


def wrap_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap radians to [-pi, pi)."""

    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def point_navigation_reference(
    robot_position_w: torch.Tensor,
    robot_yaw_w: torch.Tensor,
    target_position_w: torch.Tensor,
    completion_radius_m: float = 0.75,
    cruise_height_m: float = 10.0,
    inspection_height_m: float = 2.0,
    descent_radius_m: float = 5.0,
) -> MissionReference:
    """Create a reference that follows the current position of a moving target."""

    horizontal = torch.linalg.vector_norm(target_position_w[:, :2] - robot_position_w[:, :2], dim=-1)
    blend = (horizontal / descent_radius_m).clamp(0.0, 1.0)
    goal_position_w = target_position_w.clone()
    goal_position_w[:, 2] = target_position_w[:, 2] + torch.lerp(
        torch.full_like(blend, inspection_height_m), torch.full_like(blend, cruise_height_m), blend
    )
    distance = torch.linalg.vector_norm(goal_position_w - robot_position_w, dim=-1)
    done = distance < completion_radius_m
    phase = torch.where(
        done,
        torch.full_like(robot_yaw_w, int(MissionPhase.DONE), dtype=torch.long),
        torch.full_like(robot_yaw_w, int(MissionPhase.APPROACH), dtype=torch.long),
    )
    bearing = torch.atan2(
        target_position_w[:, 1] - robot_position_w[:, 1],
        target_position_w[:, 0] - robot_position_w[:, 0],
    )
    return MissionReference(
        position_w=goal_position_w,
        yaw_w=torch.where(done, robot_yaw_w, bearing),
        phase=phase,
        target_valid=torch.zeros_like(done),
        recording=torch.zeros_like(done),
        arc_progress=torch.zeros_like(robot_yaw_w),
    )


def minimum_scene_clearance(
    robot_position_w: torch.Tensor,
    hazard_positions_w: torch.Tensor,
    hazard_active: torch.Tensor,
    crowd_positions_w: torch.Tensor,
) -> torch.Tensor:
    """Return clearance to every active hazard and every person."""

    hazard_distance = torch.linalg.vector_norm(
        hazard_positions_w - robot_position_w[:, None, :], dim=-1
    )
    hazard_distance = torch.where(hazard_active, hazard_distance - 0.5, torch.inf)
    crowd_distance = torch.linalg.vector_norm(
        crowd_positions_w - robot_position_w[:, None, :], dim=-1
    ) - 0.4
    return torch.minimum(hazard_distance.min(dim=1).values, crowd_distance.min(dim=1).values)


class InspectionMissionSupervisor:
    """Batch mission state machine for ordered GPS seeds and detected casualties."""

    def __init__(
        self,
        num_envs: int,
        max_targets: int,
        device: torch.device | str,
        cfg: MissionSupervisorCfg | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.max_targets = max_targets
        self.device = torch.device(device)
        self.cfg = cfg or MissionSupervisorCfg()

        self.phase = torch.full((num_envs,), int(MissionPhase.APPROACH), dtype=torch.long, device=self.device)
        self.target_index = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.selected_track = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.phase_time = torch.zeros(num_envs, device=self.device)
        self.retry_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.completed_targets = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.failed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._last_target_position = torch.zeros(num_envs, 3, device=self.device)
        self._last_target_valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._arc_center_angle = torch.zeros(num_envs, device=self.device)
        self.terminal_phase = int(MissionPhase.RECORD)
        self._search_angle = torch.zeros(num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset all or selected environments."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.phase[env_ids] = int(MissionPhase.APPROACH)
        self.target_index[env_ids] = 0
        self.selected_track[env_ids] = -1
        self.phase_time[env_ids] = 0.0
        self.retry_count[env_ids] = 0
        self.completed_targets[env_ids] = 0
        self.failed[env_ids] = False
        self._last_target_position[env_ids] = 0.0
        self._last_target_valid[env_ids] = False
        self._arc_center_angle[env_ids] = 0.0
        self._search_angle[env_ids] = 0.0

    def _transition(self, mask: torch.Tensor, phase: MissionPhase) -> None:
        self.phase[mask] = int(phase)
        self.phase_time[mask] = 0.0

    def _current_gps(
        self,
        gps_targets_w: torch.Tensor,
        target_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe_count = target_counts.clamp_min(1)
        index = torch.minimum(self.target_index, safe_count - 1)
        env = torch.arange(self.num_envs, device=self.device)
        return gps_targets_w[env, index], self.target_index < target_counts

    def _select_nearest_track(
        self,
        robot_position_w: torch.Tensor,
        gps_w: torch.Tensor,
        track_positions_w: torch.Tensor,
        track_confidence: torch.Tensor,
        track_age_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match target bearings."""

        valid = (track_confidence > 0.25) & (track_age_s <= self.cfg.max_tracking_age_s)
        to_gps = gps_w[:, :2] - robot_position_w[:, :2]
        to_track = track_positions_w[..., :2] - robot_position_w[:, None, :2]
        error = wrap_pi(
            torch.atan2(to_track[..., 1], to_track[..., 0])
            - torch.atan2(to_gps[:, 1], to_gps[:, 0])[:, None]
        ).abs()
        distance = torch.where(valid, error, torch.full_like(error, torch.inf))
        selected = torch.argmin(distance, dim=1)
        selected_valid = valid.any(dim=1)
        selected = torch.where(selected_valid, selected, torch.full_like(selected, -1))
        return selected, selected_valid

    def _gather_track(self, values: torch.Tensor, default: float = 0.0) -> torch.Tensor:
        index = self.selected_track.clamp_min(0)
        env = torch.arange(self.num_envs, device=self.device)
        gathered = values[env, index]
        valid = self.selected_track >= 0
        while valid.ndim < gathered.ndim:
            valid = valid.unsqueeze(-1)
        return torch.where(valid, gathered, torch.full_like(gathered, default))

    def _servo_height(self, robot_position_w: torch.Tensor, image_y: torch.Tensor) -> torch.Tensor:
        """Height that puts the tracked box at image centre, above a detection floor."""

        half_fov = 0.5 * math.radians(self.cfg.vertical_fov_deg)
        elevation = (-(image_y - 0.5) * 2.0 * half_fov).clamp(-half_fov, half_fov)
        height = robot_position_w[:, 2] + self.cfg.standoff_m * torch.tan(elevation)
        return height.clamp_min(self.cfg.min_inspection_height_m)

    def update(
        self,
        dt: float,
        robot_position_w: torch.Tensor,
        robot_yaw_w: torch.Tensor,
        gps_targets_w: torch.Tensor,
        target_counts: torch.Tensor,
        track_positions_w: torch.Tensor,
        track_confidence: torch.Tensor,
        track_age_s: torch.Tensor,
        face_visible: torch.Tensor,
        track_image_y: torch.Tensor,
    ) -> MissionReference:
        """Advance mission state and return the current local-motion reference."""

        self.phase_time += dt
        gps_w, gps_valid = self._current_gps(gps_targets_w, target_counts)
        phase = self.phase

        approach = phase == int(MissionPhase.APPROACH)
        at_region = torch.linalg.vector_norm(robot_position_w[:, :2] - gps_w[:, :2], dim=-1) <= (
            self.cfg.gps_region_radius_m
        )
        self._transition(approach & at_region & gps_valid, MissionPhase.OBSERVE)
        self._transition(approach & ~gps_valid, MissionPhase.DONE)

        observe = self.phase == int(MissionPhase.OBSERVE)
        observation_complete = observe & (self.phase_time >= self.cfg.observation_s)
        selected, selected_valid = self._select_nearest_track(
            robot_position_w, gps_w, track_positions_w, track_confidence, track_age_s
        )
        choose = observation_complete & selected_valid
        self.selected_track[choose] = selected[choose]
        self._transition(choose, MissionPhase.SEARCH_FACE)

        selected_position = self._gather_track(track_positions_w)
        selected_confidence = self._gather_track(track_confidence)
        selected_age = self._gather_track(track_age_s, default=torch.inf)
        selected_face = self._gather_track(face_visible.float()).bool()
        selected_image_y = self._gather_track(track_image_y, default=0.5)
        selected_fresh = (
            (self.selected_track >= 0) & (selected_confidence > 0.25) & (selected_age <= self.cfg.max_tracking_age_s)
        )
        self._last_target_position = torch.where(selected_fresh[:, None], selected_position, self._last_target_position)
        self._last_target_valid |= selected_fresh

        search = self.phase == int(MissionPhase.SEARCH_FACE)
        entered_stage = search & selected_face & selected_fresh
        self._transition(entered_stage, MissionPhase.STAGE)
        
        search_failed = search & (self.phase_time >= self.cfg.search_timeout_s)
        self.retry_count[search_failed] += 1
        exhausted = search_failed & (self.retry_count >= self.cfg.max_retries)
        retry = search_failed & ~exhausted
        self.selected_track[retry | exhausted] = -1
        self._transition(retry, MissionPhase.OBSERVE)
        self.failed[exhausted] = True
        self._transition(exhausted, MissionPhase.DONE)

        target_w = self._last_target_position
        target_to_robot = robot_position_w[:, :2] - target_w[:, :2]
        current_center_angle = torch.atan2(target_to_robot[:, 1], target_to_robot[:, 0])
        self._arc_center_angle[entered_stage] = current_center_angle[entered_stage]
        center_angle = self._arc_center_angle
        half_angle = math.radians(self.cfg.arc_half_angle_deg)

        arc_progress = (self.phase_time / self.cfg.arc_duration_s).clamp(0.0, 1.0)
        inspection_angle = center_angle - half_angle
        recording = self.phase == int(MissionPhase.RECORD)
        inspection_angle = torch.where(
            recording, center_angle - half_angle + 2.0 * half_angle * arc_progress, inspection_angle
        )

        reference_position = target_w.clone()
        reference_position[:, 0] += self.cfg.standoff_m * torch.cos(inspection_angle)
        reference_position[:, 1] += self.cfg.standoff_m * torch.sin(inspection_angle)
        reference_position[:, 2] = self._servo_height(robot_position_w, selected_image_y)
        reference_yaw = wrap_pi(inspection_angle + math.pi)

        stage = self.phase == int(MissionPhase.STAGE)
        stage_position_error = torch.linalg.vector_norm(robot_position_w - reference_position, dim=-1)
        stage_yaw_error = torch.abs(wrap_pi(robot_yaw_w - reference_yaw))
        staged = (
            stage
            & selected_fresh
            & selected_face
            & (stage_position_error <= self.cfg.stage_position_tolerance_m)
            & (stage_yaw_error <= self.cfg.stage_yaw_tolerance_rad)
        )
        self._transition(staged, MissionPhase.RECORD)

        recording = self.phase == int(MissionPhase.RECORD)
        record_position_error = torch.linalg.vector_norm(robot_position_w - reference_position, dim=-1)
        record_valid = selected_fresh & selected_face & (record_position_error <= self.cfg.record_position_tolerance_m)
        invalid_record = recording & ~record_valid
        self.retry_count[invalid_record] += 1
        retry_record = invalid_record & (self.retry_count < self.cfg.max_retries)
        abandon_record = invalid_record & ~retry_record
        self._transition(retry_record, MissionPhase.STAGE)
        self.failed[abandon_record] = True
        self._transition(abandon_record, MissionPhase.DONE)
        finished_record = recording & record_valid & (self.phase_time >= self.cfg.arc_duration_s)
        self._transition(finished_record, MissionPhase.COMPLETE)

        beyond = (self.phase > self.terminal_phase) & (self.phase < int(MissionPhase.COMPLETE))
        self._transition(beyond, MissionPhase.COMPLETE)

        complete = self.phase == int(MissionPhase.COMPLETE)
        if complete.any():
            self.completed_targets[complete] += 1
            self.target_index[complete] += 1
            self.retry_count[complete] = 0
            self.selected_track[complete] = -1
            self._last_target_valid[complete] = False
            more = complete & (self.target_index < target_counts)
            self._transition(more, MissionPhase.APPROACH)
            self._transition(complete & ~more, MissionPhase.DONE)

        non_inspection = self.phase <= int(MissionPhase.OBSERVE)
        approach_position = gps_w.clone()
        approach_position[:, 2] = torch.maximum(gps_w[:, 2], robot_position_w[:, 2])
        reference_position = torch.where(non_inspection[:, None], approach_position, reference_position)
        bearing_to_gps = torch.atan2(gps_w[:, 1] - robot_position_w[:, 1], gps_w[:, 0] - robot_position_w[:, 0])
        reference_yaw = torch.where(non_inspection, bearing_to_gps, reference_yaw)

        search = self.phase == int(MissionPhase.SEARCH_FACE)
        advanced_angle = wrap_pi(self._search_angle + self.cfg.search_orbit_rate_rad_s * dt)
        self._search_angle = torch.where(
            choose,
            current_center_angle,
            torch.where(search, advanced_angle, self._search_angle),
        )
        search_position = target_w.clone()
        search_position[:, 0] += self.cfg.standoff_m * torch.cos(self._search_angle)
        search_position[:, 1] += self.cfg.standoff_m * torch.sin(self._search_angle)
        search_position[:, 2] = self._servo_height(robot_position_w, selected_image_y)
        reference_position = torch.where(search[:, None], search_position, reference_position)
        reference_yaw = torch.where(search, wrap_pi(self._search_angle + math.pi), reference_yaw)

        done = self.phase == int(MissionPhase.DONE)
        reference_position = torch.where(done[:, None], robot_position_w, reference_position)
        reference_yaw = torch.where(done, robot_yaw_w, reference_yaw)

        return MissionReference(
            position_w=reference_position,
            yaw_w=wrap_pi(reference_yaw),
            phase=self.phase.clone(),
            target_valid=self._last_target_valid.clone(),
            recording=self.phase == int(MissionPhase.RECORD),
            arc_progress=torch.where(
                self.phase == int(MissionPhase.RECORD),
                (self.phase_time / self.cfg.arc_duration_s).clamp(0.0, 1.0),
                torch.zeros_like(self.phase_time),
            ),
        )
