"""Environment that runs perception and the mission supervisor for building inspection."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat

from ..inspection import (
    PHASE_COUNT,
    BuildingMissionSupervisor,
    MissionPhase,
    MissionReference,
    OpeningStatus,
    ProposalSearchStatus,
    wrap_pi,
)
from ..perception import (
    CameraIntrinsics,
    DisparityMonitor,
    DoorwayDetector,
    PromptBoxTrackers,
    TofSampler,
    YoloePromptDetector,
    box_center_angular_error,
    fit_vertical_plane,
    project_depth_points_between_cameras,
    rays_from_normalised_points,
    rotate_to_world,
    select_active_box,
)
from .proposals import load_opening_proposals, to_environment_local, validate_association_residual
from .reward_accounting import split_reward_totals
from .truth import ledger_truth_matches

YAW_REVERSAL_DEADBAND = 0.05

FACADE_HINT_AGREEMENT = 0.95


class BuildingInspectionEnv(ManagerBasedRLEnv):
    """Model-in-loop facade survey with a deterministic mission supervisor."""

    cfg: object

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        cfg.maximum_association_residual_m = validate_association_residual(
            cfg.maximum_association_residual_m
        )
        if cfg.proposal_map_version < 0:
            raise ValueError("proposal_map_version must be non-negative")
        proposal_document = (
            load_opening_proposals(cfg.opening_proposal_path)
            if cfg.opening_proposal_path is not None
            else None
        )
        device = torch.device(cfg.sim.device)
        num_envs = cfg.scene.num_envs

        if cfg.detector_backend == "doorway":
            self._detector = DoorwayDetector(
                device=cfg.detector_device,
                image_size=cfg.detector_image_size,
                confidence=cfg.detector_confidence,
            )
        else:
            self._detector = YoloePromptDetector(
                model_path=cfg.yoloe_model_path,
                device=cfg.detector_device,
                image_size=cfg.detector_image_size,
                confidence=cfg.detector_confidence,
            )
        self._trackers = PromptBoxTrackers(num_envs, device)
        self._tof = TofSampler(num_envs, device, cfg.tof)
        self._disparity = DisparityMonitor(num_envs, device, cfg.disparity)
        if cfg.entry_training.enabled:
            cfg.mission.commit_without_dwell = True
            cfg.mission.hold_active_target = cfg.entry_training.hold_selected_opening
        self._supervisor = BuildingMissionSupervisor(num_envs, device, cfg.mission)

        self.entry_target_w = torch.zeros(num_envs, 3, device=device)
        self.entry_target_index = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.entry_start_w = torch.zeros(num_envs, 3, device=device)
        self.episode_records: list[dict] = []
        self.entry_start_width = cfg.entry_training.initial_width
        self.unauthorized_crossing_now = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._entry_crossing_depth_m = torch.zeros(num_envs, device=device)
        self._entry_closest_clearance_m = torch.full((num_envs,), cfg.max_clearance_m, device=device)
        self._entry_visible_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._entry_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._entry_reacquired = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._phase_steps = torch.zeros(num_envs, PHASE_COUNT, dtype=torch.long, device=device)
        self._transit_arrived = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._min_reference_distance = torch.full((num_envs,), torch.inf, device=device)
        self._final_reference_distance = torch.zeros(num_envs, device=device)
        self._detector_proposals = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._proposal_search_first_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._proposal_ledger_first_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._proposal_match_first_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._proposal_direct_reference = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._validation_target_region_missing = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._validation_target_projected = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._validation_opening_measured = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._in_validation = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._validation_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._selected_truth_gap = torch.zeros(num_envs, device=device)
        self._selected_gap_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._selected_true_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._selected_false_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._selected_true_confidence = torch.zeros(num_envs, device=device)
        self._selected_false_confidence = torch.zeros(num_envs, device=device)
        self._selected_true_hits = torch.zeros(num_envs, device=device)
        self._selected_false_hits = torch.zeros(num_envs, device=device)
        self._refresh_attempted = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._refresh_found = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._refresh_no_candidate = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._refresh_ambiguous = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._validation_tracks_any = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._validation_tracks_confirmed = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._validation_tracks_fresh = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._assoc_stored_box_zero = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._assoc_box_gap_sum = torch.zeros(num_envs, device=device)
        self._assoc_box_gap_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._assoc_stored_id_absent = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._assoc_stored_id_stale = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._assoc_no_tracks = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._assoc_no_identity = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._assoc_recovered = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._assoc_by_identity = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._unauthorized_crossing_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._crossing_scored_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._crossing_observable_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._inside_unauthorized_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._was_behind_facade = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._max_facade_depth_m = torch.zeros(num_envs, device=device)
        self._facade_graze_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._crossed_inward_near = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._crossed_inward_far = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._observe_run_s = torch.zeros(num_envs, device=device)
        self._was_observing = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._min_observe_hold_s = torch.full((num_envs,), torch.inf, device=device)
        self._unauthorized_phase = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._validation_box_available = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._validation_identity_correct = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._previous_phase = torch.full(
            (num_envs,), int(MissionPhase.TRANSIT), dtype=torch.long, device=device
        )
        self._survey_station_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._survey_travel_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._since_survey_s = torch.zeros(num_envs, device=device)
        self._survey_to_entry_sum_s = torch.zeros(num_envs, device=device)
        self._survey_to_entry_max_s = torch.zeros(num_envs, device=device)
        self._survey_to_entry_events = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._entry_events = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._repeat_entries = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._entered_slot = torch.zeros(
            num_envs, self._supervisor.ledger.max_openings, dtype=torch.bool, device=device
        )
        self._yaw_rate_abs_sum = torch.zeros(num_envs, device=device)
        self._yaw_rate_abs_max = torch.zeros(num_envs, device=device)
        self._yaw_reversals = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._previous_yaw_rate = torch.zeros(num_envs, device=device)
        self._previous_yaw_w = torch.zeros(num_envs, device=device)
        self._yaw_history_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._reward_previous_distance = torch.zeros(num_envs, device=device)
        self._reward_previous_reference_phase = torch.full(
            (num_envs,), int(MissionPhase.TRANSIT), dtype=torch.long, device=device
        )
        self._reward_reference_history_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._reward_previous_box_error = torch.zeros(num_envs, device=device)
        self._reward_previous_box_phase = torch.full(
            (num_envs,), int(MissionPhase.TRANSIT), dtype=torch.long, device=device
        )
        self._reward_box_history_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._reward_previous_yaw_error = torch.zeros(num_envs, device=device)
        self._reward_previous_yaw_phase = torch.full(
            (num_envs,), int(MissionPhase.TRANSIT), dtype=torch.long, device=device
        )
        self._reward_yaw_progress_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._reward_previous_yaw_rate = torch.zeros(num_envs, device=device)
        self._reward_yaw_history_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._previous_transit_completion = torch.zeros(num_envs, device=device)
        self._previous_station = torch.zeros(num_envs, device=device)
        self._previous_observing = torch.zeros(num_envs, device=device)
        self._previous_consumed = torch.zeros(num_envs, device=device)
        self._previous_inspected = torch.zeros(num_envs, device=device)
        self._previous_safe_exit = torch.zeros(num_envs, device=device)
        self._previous_floor = torch.zeros(num_envs, device=device)
        self._previous_entry_crossing = torch.zeros(num_envs, device=device)
        self._previous_wrong_opening = torch.zeros(num_envs, device=device)
        self._observe_yaw_abs_sum = torch.zeros(num_envs, device=device)
        self._observe_yaw_abs_max = torch.zeros(num_envs, device=device)
        self._observe_yaw_reversals = torch.zeros(num_envs, dtype=torch.long, device=device)

        self.building_standoff_w = torch.zeros(num_envs, 3, device=device)
        self.opening_truth_w = torch.zeros(num_envs, cfg.max_truth_openings, 3, device=device)
        self.truth_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.tof_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.active_opening_box = torch.zeros(num_envs, 4, device=device)
        self.active_opening_box_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.active_opening_box_centered = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.active_opening_confidence = torch.zeros(num_envs, device=device)
        self.active_opening_age_s = torch.zeros(num_envs, device=device)
        self.active_target_changed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._previous_active_slot = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.tof_clear = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.tof_range_m = torch.zeros(num_envs, device=device)
        self.tof_point_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.tof_threshold_configured = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.clearance_m = torch.full((num_envs,), cfg.max_clearance_m, device=device)
        self.clearance_known = torch.zeros(num_envs, dtype=torch.bool, device=device)

        self._detector_elapsed_s = cfg.detector_period_s
        self._detector_ran = False
        self._mission_step = -1
        self._reference = MissionReference(
            position_w=torch.zeros(num_envs, 3, device=device),
            yaw_w=torch.zeros(num_envs, device=device),
            phase=torch.full((num_envs,), int(MissionPhase.TRANSIT), dtype=torch.long, device=device),
            entry_authorized=torch.zeros(num_envs, dtype=torch.bool, device=device),
            crossing_opening=torch.zeros(num_envs, dtype=torch.bool, device=device),
            target_valid=torch.zeros(num_envs, dtype=torch.bool, device=device),
            detector_enabled=torch.zeros(num_envs, dtype=torch.bool, device=device),
            accepts_proposals=torch.zeros(num_envs, dtype=torch.bool, device=device),
            observation_progress=torch.zeros(num_envs, device=device),
        )

        self._camera_intrinsics: CameraIntrinsics | None = None
        self._tof_camera_intrinsics: CameraIntrinsics | None = None
        super().__init__(cfg, render_mode, **kwargs)
        if proposal_document is not None:
            proposal_positions = to_environment_local(
                proposal_document.position_w,
                self.scene.env_origins,
            )
            self._supervisor.proposal_search.replace(
                proposal_document.ids,
                proposal_positions,
                map_version=cfg.proposal_map_version,
            )

    @property
    def mission(self) -> BuildingMissionSupervisor:
        self._refresh()
        return self._supervisor

    @property
    def mission_reference(self) -> MissionReference:
        self._refresh()
        return self._reference

    @property
    def disparity(self) -> DisparityMonitor:
        self._refresh()
        return self._disparity

    @property
    def trackers(self) -> PromptBoxTrackers:
        self._refresh()
        return self._trackers

    @property
    def robot_yaw_w(self) -> torch.Tensor:
        _, _, yaw = euler_xyz_from_quat(self.scene["robot"].data.root_quat_w)
        return yaw

    @property
    def true_solved_openings(self) -> torch.Tensor:
        """Solved detector-ledger entries that match simulator truth."""

        ledger = self._supervisor.ledger
        return ledger_truth_matches(
            ledger.position_w,
            ledger.solved,
            self.opening_truth_w,
            self.truth_count,
            self.cfg.truth_match_radius_m,
        )

    @property
    def true_recorded_openings(self) -> torch.Tensor:
        """Ledger entries fit to approach that match simulator truth."""

        ledger = self._supervisor.ledger
        return ledger_truth_matches(
            ledger.position_w,
            ledger.approachable,
            self.opening_truth_w,
            self.truth_count,
            self.cfg.truth_match_radius_m,
        )

    @property
    def true_inspected_openings(self) -> torch.Tensor:
        """Safely inspected detector-ledger entries that match simulator truth."""

        ledger = self._supervisor.ledger
        inspected = ledger.status == int(OpeningStatus.INSPECTED)
        return ledger_truth_matches(
            ledger.position_w,
            inspected,
            self.opening_truth_w,
            self.truth_count,
            self.cfg.truth_match_radius_m,
        )

    @property
    def true_validated_openings(self) -> torch.Tensor:
        """ToF-cleared detector-ledger entries that match simulator truth."""

        ledger = self._supervisor.ledger
        return ledger_truth_matches(
            ledger.position_w,
            ledger.validated,
            self.opening_truth_w,
            self.truth_count,
            self.cfg.truth_match_radius_m,
        )

    def _within_selected_opening_span(self, robot_w: torch.Tensor) -> torch.Tensor:
        """Whether the aircraft is over the stretch of wall the episode selected."""

        hint = self._training_facade_normal
        if hint is None or not self.cfg.entry_training.enabled:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        tangent = torch.stack((-hint[:, 1], hint[:, 0], torch.zeros_like(hint[:, 0])), dim=-1)
        offset = robot_w - self.entry_target_w
        lateral = (offset * tangent).sum(dim=-1).abs()
        vertical = offset[:, 2].abs()
        span = self.cfg.crossing_span_m
        return (lateral <= span) & (vertical <= span)

    def _crossing_plane(
        self, fitted_normal: torch.Tensor, fitted_offset: torch.Tensor, fitted_valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the plane a crossing is measured against, and whether it is known."""

        hint = self._training_facade_normal
        if hint is None:
            return fitted_normal, fitted_offset, fitted_valid
        if self.cfg.entry_training.enabled:
            return hint, (hint * self.entry_target_w).sum(dim=-1), torch.ones_like(fitted_valid)
        aligned = (
            self._supervisor.facade_normal_w * hint
        ).sum(dim=-1) >= FACADE_HINT_AGREEMENT
        return hint, self._supervisor.facade_offset_m, self._supervisor.facade_known & aligned

    @property
    def entry_crossing(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return whether the aircraft is fully through the chosen opening, or another one."""

        if self.cfg.entry_training.enabled:
            fully = self._entry_crossing_depth_m >= self.cfg.entry_training.success_crossing_margin_m
            return self._crossed_inward_near & fully, self._crossed_inward_far & fully
        margin = self.cfg.entry_training.success_crossing_margin_m
        robot = self.scene["robot"].data.root_pos_w - self.scene.env_origins
        ledger = self._supervisor.ledger
        active_slot = self._supervisor.active_slot
        env_index = torch.arange(self.num_envs, device=self.device)
        target = ledger.position_w[env_index, active_slot.clamp_min(0)]
        through = self._entry_crossing_depth_m >= margin
        near_active = (active_slot >= 0) & (
            torch.linalg.vector_norm(robot - target, dim=-1) <= self.cfg.truth_match_radius_m
        )
        return through & near_active, through & (active_slot >= 0) & ~near_active

    @property
    def facade_seed_error(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Angle and offset error of the working facade plane against the truth openings."""

        known = torch.arange(self.opening_truth_w.shape[1], device=self.device)[None, :] < self.truth_count[:, None]
        prior = self._supervisor.facade_normal_w
        truth_normal, truth_offset, usable = fit_vertical_plane(self.opening_truth_w, known, prior)
        working_normal = self._supervisor.facade_normal_w
        cosine = (working_normal * truth_normal).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.rad2deg(torch.arccos(cosine))
        offset = (self._supervisor.facade_offset_m - truth_offset).abs()
        on_briefed_face = usable & (self._supervisor.facade_index == 0)
        return angle, offset, on_briefed_face

    @property
    def true_occupied_openings(self) -> torch.Tensor:
        """Occupied ledger entries that match a real opening, whatever their state."""

        ledger = self._supervisor.ledger
        return ledger_truth_matches(
            ledger.position_w,
            ledger.occupied,
            self.opening_truth_w,
            self.truth_count,
            self.cfg.truth_match_radius_m,
        )

    @property
    def selected_target_is_true(self) -> torch.Tensor:
        """Whether the ledger entry now selected for approach is a real opening."""

        ledger = self._supervisor.ledger
        active_slot = self._supervisor.active_slot
        selected = torch.zeros_like(ledger.occupied)
        env_index = torch.arange(active_slot.shape[0], device=self.device)
        selected[env_index, active_slot.clamp_min(0)] = active_slot >= 0
        return ledger_truth_matches(
            ledger.position_w,
            selected,
            self.opening_truth_w,
            self.truth_count,
            self.cfg.truth_match_radius_m,
        ).any(dim=1)

    @property
    def _intrinsics(self) -> CameraIntrinsics:
        """Read the rendering camera intrinsics once."""

        if self._camera_intrinsics is None:
            camera = self.scene["rgb_camera"]
            if not camera.cfg.update_latest_camera_pose:
                raise RuntimeError(
                    "rgb_camera must set update_latest_camera_pose=True. Without it Isaac Lab "
                    "freezes pos_w and quat_w_ros at their startup values, and every opening "
                    "bearing is anchored at the wrong place."
                )
            matrix = camera.data.intrinsic_matrices[0]
            height, width = camera.image_shape
            self._camera_intrinsics = CameraIntrinsics(
                fx=float(matrix[0, 0]),
                fy=float(matrix[1, 1]),
                cx=float(matrix[0, 2]),
                cy=float(matrix[1, 2]),
                width=int(width),
                height=int(height),
            )
        return self._camera_intrinsics

    @property
    def _tof_intrinsics(self) -> CameraIntrinsics:
        """Intrinsics and live pose contract for the ToF camera."""

        if self._tof_camera_intrinsics is None:
            camera = self.scene["tof_camera"]
            if not camera.cfg.update_latest_camera_pose:
                raise RuntimeError("tof_camera must set update_latest_camera_pose=True for active-opening projection.")
            matrix = camera.data.intrinsic_matrices[0]
            height, width = camera.image_shape
            self._tof_camera_intrinsics = CameraIntrinsics(
                fx=float(matrix[0, 0]),
                fy=float(matrix[1, 1]),
                cx=float(matrix[0, 2]),
                cy=float(matrix[1, 2]),
                width=int(width),
                height=int(height),
            )
        return self._tof_camera_intrinsics

    @property
    def nearest_person_range_m(self) -> torch.Tensor:
        """Range to the closest tracked person from box height, or inf when none is tracked."""

        self._refresh()
        people = self._trackers.people
        pixels = people.state[..., 3].clamp_min(1e-3) * self._intrinsics.height
        ranges = (self.cfg.nominal_person_height_m * self._intrinsics.fy / pixels).clamp(0.5, 25.0)
        return torch.where(people.confirmed(), ranges, torch.inf).min(dim=1).values

    def _refresh(self) -> None:
        """Run perception and the mission once per environment step."""

        if self._mission_step == self.common_step_counter:
            return
        self._mission_step = self.common_step_counter
        self._step_perception()
        self._step_mission()

    def _step_perception(self) -> None:
        dt = self.step_dt
        self._detector_elapsed_s += dt
        detector_due = self._detector_elapsed_s + 1.0e-9 >= self.cfg.detector_period_s
        detector_enabled = self._reference.detector_enabled
        self._detector_ran = bool(detector_due and detector_enabled.any().item())

        points = self._tof.sample(self.scene["tof_camera"].data.output["distance_to_image_plane"])
        clearance_valid, _ = self._tof.corridor_evidence(points)
        tof_clearance_m = self._tof.corridor_clearance(points)
        self._tof_points = points
        far = torch.full_like(tof_clearance_m, self.cfg.max_clearance_m)
        self.clearance_known = clearance_valid
        self.clearance_m = torch.where(clearance_valid, tof_clearance_m, far)

        rgb = self.scene["rgb_camera"].data.output["rgb"]
        self._disparity.tick(dt, rgb)

        if not self._detector_ran:
            self._trackers.predict(dt)
        else:
            self._detector_elapsed_s %= self.cfg.detector_period_s
            detections = self._detector.infer_batch(rgb)
            detections.valid &= detector_enabled[:, None].to(detections.valid.device)
            self._trackers.step(dt, detections)

        target_box, target_region = self._active_opening_rgb_region()
        aim_error = box_center_angular_error(target_box, self._intrinsics)
        self.active_opening_box_centered = target_region & (
            aim_error.abs() <= self.cfg.mission.entry_aim_tolerance_rad
        ).all(dim=-1)
        tof_camera = self.scene["tof_camera"]
        rgb_camera = self.scene["rgb_camera"]
        projected_xyn, _, projected_valid = project_depth_points_between_cameras(
            points_xyn=points.points_xyn,
            depth_m=points.range_m,
            point_valid=points.valid,
            source_intrinsics=self._tof_intrinsics,
            source_position_w=tof_camera.data.pos_w - self.scene.env_origins,
            source_quat_w_ros=tof_camera.data.quat_w_ros,
            target_intrinsics=self._intrinsics,
            target_position_w=rgb_camera.data.pos_w - self.scene.env_origins,
            target_quat_w_ros=rgb_camera.data.quat_w_ros,
        )
        evidence = self._tof.target_evidence(
            points, projected_xyn, projected_valid, target_box, target_region
        )
        self.tof_valid = evidence.valid
        self.tof_clear = evidence.clear
        self.tof_range_m = evidence.opening_depth_m
        self.tof_point_count = evidence.point_count
        self.tof_threshold_configured = evidence.threshold_configured
        validating = self._reference.phase == int(MissionPhase.VALIDATE)
        self._in_validation = validating
        self._validation_target_region_missing += (validating & ~evidence.target_projected).long()
        self._validation_target_projected += (validating & evidence.target_projected).long()
        self._validation_opening_measured += (validating & evidence.opening_measured).long()
        self._validation_steps += validating.long()
        self._validation_box_available += (validating & target_region).long()
        self._validation_identity_correct += (validating & self.selected_target_is_true).long()
        ledger_now = self._supervisor.ledger
        active_now = self._supervisor.active_slot
        index_now = torch.arange(self.num_envs, device=self.device)
        chosen = active_now.clamp_min(0)
        selected_confidence = ledger_now.confidence[index_now, chosen]
        selected_hits = ledger_now.hits[index_now, chosen].float()
        counted = validating & (active_now >= 0)
        real = counted & self.selected_target_is_true
        false_target = counted & ~self.selected_target_is_true
        self._selected_true_steps += real.long()
        self._selected_false_steps += false_target.long()
        self._selected_true_confidence += torch.where(real, selected_confidence, torch.zeros_like(selected_confidence))
        self._selected_false_confidence += torch.where(
            false_target, selected_confidence, torch.zeros_like(selected_confidence)
        )
        self._selected_true_hits += torch.where(real, selected_hits, torch.zeros_like(selected_hits))
        truth = self.opening_truth_w
        known = torch.arange(truth.shape[1], device=self.device)[None, :] < self.truth_count[:, None]
        selected_position = ledger_now.position_w[index_now, chosen]
        gap = torch.linalg.vector_norm(selected_position[:, None, :] - truth, dim=-1)
        nearest_truth = gap.masked_fill(~known, torch.inf).amin(dim=1)
        finite_gap = torch.where(torch.isfinite(nearest_truth), nearest_truth, torch.zeros_like(nearest_truth))
        self._selected_truth_gap += torch.where(counted, finite_gap, torch.zeros_like(finite_gap))
        self._selected_gap_steps += counted.long()
        self._selected_false_hits += torch.where(false_target, selected_hits, torch.zeros_like(selected_hits))

    def _active_opening_rgb_region(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the live RGB box associated with the active ledger opening."""

        ledger = self._supervisor.ledger
        active_slot = self._supervisor.active_slot
        slot = active_slot.clamp_min(0)
        env_index = torch.arange(active_slot.shape[0], device=self.device)
        target_valid = active_slot >= 0

        openings = self._trackers.openings
        fresh = openings.confirmed() & (
            openings.age_s <= self.cfg.track_prediction_grace_s + self.step_dt + 1.0e-6
        )
        any_track = fresh.any(dim=1)
        identity_present = (fresh & (openings.track_id == ledger.track_id[env_index, slot][:, None])).any(dim=1)
        rgb_box, associated = select_active_box(
            openings.state[..., :4],
            openings.track_id,
            fresh,
            active_track_id=ledger.track_id[env_index, slot],
            stored_box_xywhn=ledger.box_xywhn[env_index, slot],
            target_valid=target_valid,
        )
        self.active_opening_box = rgb_box
        self.active_opening_box_valid = target_valid & associated
        box_gap = torch.linalg.vector_norm(openings.state[..., :4] - rgb_box[:, None, :], dim=-1)
        selected_track = box_gap.masked_fill(~fresh, torch.inf).argmin(dim=1)
        selected_confidence = openings.confidence[env_index, selected_track]
        selected_age = openings.age_s[env_index, selected_track]
        self.active_opening_confidence = torch.where(
            self.active_opening_box_valid, selected_confidence, torch.zeros_like(selected_confidence)
        )
        self.active_opening_age_s = torch.where(
            self.active_opening_box_valid, selected_age, torch.zeros_like(selected_age)
        )
        self._entry_visible_steps += (target_valid & associated).long()
        self._entry_reacquired += (target_valid & associated & ~identity_present).long()
        wanted = target_valid & self._in_validation
        stored_id = ledger.track_id[env_index, slot]
        stored_box = ledger.box_xywhn[env_index, slot]
        self._assoc_stored_box_zero += (wanted & (stored_box.abs().sum(dim=-1) == 0.0)).long()
        gap = torch.linalg.vector_norm(openings.state[..., :2] - stored_box[:, None, :2], dim=-1)
        nearest_gap = gap.masked_fill(~fresh, torch.inf).amin(dim=1)
        self._assoc_box_gap_sum += torch.where(wanted & any_track, nearest_gap, torch.zeros_like(nearest_gap))
        self._assoc_box_gap_count += (wanted & any_track).long()
        self._assoc_stored_id_absent += (wanted & (stored_id < 0)).long()
        self._assoc_stored_id_stale += (wanted & (stored_id >= 0) & ~identity_present).long()
        openings_raw = self._trackers.openings
        self._refresh_attempted += (wanted & ledger.refresh_attempted).long()
        self._refresh_found += (wanted & self._supervisor.active_refreshed).long()
        self._refresh_no_candidate += (wanted & ledger.refresh_no_candidate).long()
        self._refresh_ambiguous += (wanted & ledger.refresh_ambiguous).long()
        self._validation_tracks_any += (wanted & openings_raw.valid.any(dim=1)).long()
        self._validation_tracks_confirmed += (wanted & openings_raw.confirmed().any(dim=1)).long()
        self._validation_tracks_fresh += (wanted & any_track).long()
        self._assoc_no_tracks += (wanted & ~any_track).long()
        self._assoc_no_identity += (wanted & any_track & ~identity_present).long()
        self._assoc_recovered += (wanted & ~identity_present & associated).long()
        self._assoc_by_identity += (wanted & identity_present & associated).long()
        return rgb_box, target_valid & associated

    def _angular_sizes(self) -> torch.Tensor:
        """Angular width and height of each opening box, from the intrinsics."""

        boxes = self._trackers.openings.state[..., 2:4]
        return 2.0 * torch.atan(
            0.5
            * boxes
            * torch.tensor(
                [self._intrinsics.width / self._intrinsics.fx, self._intrinsics.height / self._intrinsics.fy],
                device=self.device,
            )
        )

    def _opening_rays(self) -> tuple[torch.Tensor, ...]:
        """World rays through each opening track."""

        openings = self._trackers.openings
        centres = openings.state[..., :2]
        camera = self.scene["rgb_camera"]

        rays = rays_from_normalised_points(centres, self._intrinsics)
        rays_w = rotate_to_world(rays, camera.data.quat_w_ros)
        origin_w = camera.data.pos_w - self.scene.env_origins
        admitted = self._reference.accepts_proposals[:, None]
        return origin_w, rays_w, openings.confidence, openings.confirmed() & self._detector_ran & admitted

    @property
    def _training_facade_normal(self) -> torch.Tensor | None:
        """Return hidden facade truth only for the isolated entry curriculum."""

        if not self.cfg.entry_training.enabled:
            return None
        return torch.tensor(
            self.cfg.entry_training.truth_facade_normal_w, device=self.device
        ).expand(self.num_envs, 3)


    def _facade_plane(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fit the wall the sweep is following, from the openings already fixed on it."""

        ledger = self._supervisor.ledger
        settled = ledger.solved & ledger.occupied
        enough = settled.sum(dim=1) >= self.cfg.min_openings_for_plane
        hint = self._training_facade_normal
        if hint is not None:
            weights = settled.float()
            along = (ledger.position_w * hint[:, None, :]).sum(dim=-1)
            offset = (along * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return hint, offset, enough
        normal, offset, usable = fit_vertical_plane(ledger.position_w, settled, self._supervisor.facade_normal_w)
        return normal, offset, usable & enough

    def _step_mission(self) -> None:
        origin_w, rays_w, confidence, valid = self._opening_rays()
        openings = self._trackers.openings
        people = self._trackers.people
        robot = self.scene["robot"].data.root_pos_w - self.scene.env_origins
        plane_normal, plane_offset, plane_valid = self._facade_plane()

        self._reference = self._supervisor.update(
            self.step_dt,
            robot_position_w=robot,
            robot_yaw_w=self.robot_yaw_w,
            building_standoff_w=self.building_standoff_w,
            opening_origin_w=origin_w,
            opening_ray_w=rays_w,
            opening_angular_wh=self._angular_sizes(),
            opening_track_id=openings.track_id,
            opening_box_xywhn=openings.state[..., :4],
            opening_confidence=confidence,
            opening_valid=valid,
            opening_box_centered=self.active_opening_box_centered,
            opening_detected=openings.confirmed(),
            facade_normal_hint=self._training_facade_normal,
            person_track_id=people.track_id,
            person_box_xywhn=people.state[..., :4],
            person_confidence=people.confidence,
            person_age_s=people.age_s,
            person_valid=people.confirmed(),
            tof_valid=self.tof_valid,
            tof_clear=self.tof_clear,
            clearance_m=self.clearance_m,
            clearance_known=self.clearance_known,
            avoidance_active=self._disparity.avoidance_active,
            time_s=self.common_step_counter * self.step_dt,
            maximum_association_residual_m=self.cfg.maximum_association_residual_m,
            facade_normal_w=plane_normal,
            facade_offset_m=plane_offset,
            facade_valid=plane_valid,
        )
        ledger = self._supervisor.ledger
        detector_backed = (ledger.source == 0) & ledger.approachable
        proposal_state = self._supervisor.proposal_search.status
        searching_now = (
            self._supervisor.proposal_search.valid
            & (proposal_state != int(ProposalSearchStatus.PENDING))
        ).any(dim=1)
        detector_now = detector_backed.any(dim=1)
        episode_step = self.episode_length_buf.long()
        self._proposal_search_first_step = torch.where(
            searching_now & (self._proposal_search_first_step < 0),
            episode_step,
            self._proposal_search_first_step,
        )
        self._proposal_ledger_first_step = torch.where(
            detector_now & (self._proposal_ledger_first_step < 0),
            episode_step,
            self._proposal_ledger_first_step,
        )
        matched_now = (
            self._supervisor.proposal_search.status == int(ProposalSearchStatus.MATCHED)
        ).any(dim=1)
        self._proposal_match_first_step = torch.where(
            matched_now & (self._proposal_match_first_step < 0),
            episode_step,
            self._proposal_match_first_step,
        )
        if self._supervisor.proposal_search.proposal_count:
            reference_to_hint = torch.cdist(
                self._reference.position_w[:, None, :],
                self._supervisor.proposal_search.position_local,
            ).squeeze(1)
            approach = self._reference.phase == int(MissionPhase.APPROACH)
            self._proposal_direct_reference |= approach & (reference_to_hint < 1.0e-4).any(dim=1)
        current_active_slot = self._supervisor.active_slot
        self.active_target_changed = current_active_slot != self._previous_active_slot
        self._previous_active_slot = current_active_slot.clone()
        normal, offset, plane_known = self._crossing_plane(plane_normal, plane_offset, plane_valid)
        depth = offset - (robot * normal).sum(dim=-1)
        behind_plane = plane_known & (depth > 0.0)
        behind = plane_known & (depth > self.cfg.unauthorized_depth_m)
        self._facade_graze_steps += (behind_plane & ~behind).long()
        self._crossing_scored_steps += 1
        self._crossing_observable_steps += plane_known.long()
        near_target = self._within_selected_opening_span(robot)
        crossed_inward = behind & near_target & ~self._was_behind_facade
        self.unauthorized_crossing_now = crossed_inward & ~self._reference.entry_authorized
        self._unauthorized_crossing_steps += self.unauthorized_crossing_now.long()
        self._inside_unauthorized_steps += (behind & ~self._reference.entry_authorized).long()
        self._was_behind_facade = behind
        self._crossed_inward_near |= crossed_inward & near_target
        self._crossed_inward_far |= crossed_inward & ~near_target
        self._max_facade_depth_m = torch.maximum(
            self._max_facade_depth_m, torch.where(plane_known, depth, torch.zeros_like(depth))
        )
        first = self.unauthorized_crossing_now & (self._unauthorized_phase < 0)
        self._unauthorized_phase = torch.where(first, self._reference.phase, self._unauthorized_phase)
        self._entry_crossing_depth_m = torch.where(plane_known, depth, torch.zeros_like(depth))
        observing = self._reference.phase == int(MissionPhase.OBSERVE)
        finished_hold = self._was_observing & ~observing
        self._min_observe_hold_s = torch.where(
            finished_hold,
            torch.minimum(self._min_observe_hold_s, self._observe_run_s),
            self._min_observe_hold_s,
        )
        self._observe_run_s = torch.where(
            observing, self._observe_run_s + self.step_dt, torch.zeros_like(self._observe_run_s)
        )
        self._was_observing = observing
        self._entry_steps += 1
        self._entry_closest_clearance_m = torch.minimum(self._entry_closest_clearance_m, self.clearance_m)
        self._detector_proposals += valid.sum(dim=1)
        self._phase_steps.scatter_add_(
            1, self._reference.phase[:, None], torch.ones_like(self._reference.phase[:, None])
        )
        self._transit_arrived |= self._reference.phase != int(MissionPhase.TRANSIT)
        reference_distance = torch.linalg.vector_norm(self._reference.position_w - robot, dim=-1)
        self._final_reference_distance = reference_distance
        self._min_reference_distance = torch.minimum(self._min_reference_distance, reference_distance)
        self._record_multi_window(robot)

    def _record_multi_window(self, robot: torch.Tensor) -> None:
        """Accumulate what a multi-window mission is judged on: pace, coverage, and turning."""

        phase = self._reference.phase
        survey = phase == int(MissionPhase.SURVEY)
        self._survey_station_steps += (survey & self._supervisor.at_station).long()
        self._survey_travel_steps += (survey & ~self._supervisor.at_station).long()

        entered_survey = survey & (self._previous_phase != int(MissionPhase.SURVEY))
        self._since_survey_s = torch.where(
            entered_survey, torch.zeros_like(self._since_survey_s), self._since_survey_s + self.step_dt
        )

        entry_started = (phase == int(MissionPhase.ENTRY)) & (self._previous_phase != int(MissionPhase.ENTRY))
        self._survey_to_entry_sum_s += torch.where(
            entry_started, self._since_survey_s, torch.zeros_like(self._since_survey_s)
        )
        self._survey_to_entry_max_s = torch.maximum(
            self._survey_to_entry_max_s,
            torch.where(entry_started, self._since_survey_s, torch.zeros_like(self._since_survey_s)),
        )
        self._survey_to_entry_events += entry_started.long()
        self._entry_events += entry_started.long()

        slot = self._supervisor.active_slot.clamp_min(0)[:, None]
        entering_known = entry_started & self._entered_slot.gather(1, slot)[:, 0]
        self._repeat_entries += entering_known.long()
        self._entered_slot.scatter_(1, slot, self._entered_slot.gather(1, slot) | entry_started[:, None])

        yaw = self.robot_yaw_w
        yaw_rate = wrap_pi(yaw - self._previous_yaw_w) / self.step_dt
        yaw_rate = torch.where(self._yaw_history_valid, yaw_rate, torch.zeros_like(yaw_rate))
        magnitude = yaw_rate.abs()
        self._yaw_rate_abs_sum += magnitude
        self._yaw_rate_abs_max = torch.maximum(self._yaw_rate_abs_max, magnitude)
        moving = magnitude > YAW_REVERSAL_DEADBAND
        reversed_now = (
            moving & (self._previous_yaw_rate.abs() > YAW_REVERSAL_DEADBAND)
            & (torch.sign(yaw_rate) != torch.sign(self._previous_yaw_rate))
        )
        self._yaw_reversals += reversed_now.long()

        observing = phase == int(MissionPhase.OBSERVE)
        self._observe_yaw_abs_sum += magnitude * observing
        self._observe_yaw_abs_max = torch.maximum(
            self._observe_yaw_abs_max, torch.where(observing, magnitude, torch.zeros_like(magnitude))
        )
        self._observe_yaw_reversals += (reversed_now & observing).long()

        self._previous_yaw_rate = yaw_rate.clone()
        self._previous_yaw_w = yaw.clone()
        self._yaw_history_valid = torch.ones_like(self._yaw_history_valid)
        self._previous_phase = phase.clone()

    def _record_episode_results(
        self,
        ids: torch.Tensor,
        phase_steps: torch.Tensor,
        timeout: torch.Tensor,
        collision: torch.Tensor,
        recovery_failed: torch.Tensor,
        action,
        reward_raw: dict[str, torch.Tensor],
        reward_weighted: dict[str, torch.Tensor],
    ) -> None:
        """Record completed episodes before counter reset."""

        crossed_selected, crossed_elsewhere = self.entry_crossing
        _, facade_offset_error_all, _ = self.facade_seed_error
        hint = self._training_facade_normal
        if hint is None:
            anchor_depth = torch.zeros(self.num_envs, device=self.device)
            station_depth = anchor_depth
        else:
            plane_offset = (hint * self.entry_target_w).sum(dim=-1)
            anchor_depth = (hint * self._supervisor.sweep_anchor_w).sum(dim=-1) - plane_offset
            station_position, _ = self._supervisor.station_pose()
            station_depth = (hint * station_position).sum(dim=-1) - plane_offset
        inspected_slots = self._supervisor.ledger.status == int(OpeningStatus.INSPECTED)
        inspected = inspected_slots.any(dim=1)
        inspected_count = inspected_slots.sum(dim=1)
        survey_steps = phase_steps[:, int(MissionPhase.SURVEY)]
        entry_steps = phase_steps[:, int(MissionPhase.ENTRY)]
        observe_steps = phase_steps[:, int(MissionPhase.OBSERVE)]
        exit_steps = phase_steps[:, int(MissionPhase.EXIT)]
        for row, env_id in enumerate(ids.tolist()):
            self.episode_records.append(
                {
                    "env": env_id,
                    "rewards_raw": {name: float(value[row].item()) for name, value in reward_raw.items()},
                    "rewards_weighted": {
                        name: float(value[row].item()) for name, value in reward_weighted.items()
                    },
                    "timeout": bool(timeout[row].item()),
                    "collision": bool(collision[row].item()),
                    "recovery_failed": bool(recovery_failed[row].item()),
                    "unauthorized_crossing": bool(self._unauthorized_crossing_steps[env_id].item() > 0),
                    "crossing_scored_steps": int(self._crossing_scored_steps[env_id].item()),
                    "crossing_observable_steps": int(self._crossing_observable_steps[env_id].item()),
                    "inside_unauthorized_steps": int(self._inside_unauthorized_steps[env_id].item()),
                    "max_facade_depth_m": float(self._max_facade_depth_m[env_id].item()),
                    "facade_graze_steps": int(self._facade_graze_steps[env_id].item()),
                    "facade_offset_error_m": float(facade_offset_error_all[env_id].item()),
                    "station_standoff_m": float(self._supervisor.station_standoff_m[env_id].item()),
                    "anchor_depth_m": float(anchor_depth[env_id].item()),
                    "station_depth_m": float(station_depth[env_id].item()),
                    "unauthorized_phase": int(self._unauthorized_phase[env_id].item()),
                    "crossed_selected": bool(crossed_selected[env_id].item()),
                    "crossed_elsewhere": bool(crossed_elsewhere[env_id].item()),
                    "entered": bool(entry_steps[row].item() > 0),
                    "observed": bool(observe_steps[row].item() > 0),
                    "exited": bool(exit_steps[row].item() > 0),
                    "observe_s": float(observe_steps[row].item()) * self.step_dt,
                    "min_observe_hold_s": float(self._min_observe_hold_s[env_id].item()),
                    "observation_by_hold": bool(self._supervisor.observation_by_hold[env_id].item()),
                    "observation_by_empty": bool(self._supervisor.observation_by_empty[env_id].item()),
                    "observation_by_timeout": bool(self._supervisor.observation_by_timeout[env_id].item()),
                    "observed_hold_s": float(self._supervisor.observed_hold_s[env_id].item()),
                    "episode_steps": int(self.episode_length_buf[env_id].item()),
                    "final_reference_distance_m": float(
                        self._final_reference_distance[env_id].item()
                    ),
                    "final_position_local": [
                        float(value)
                        for value in (
                            self.scene["robot"].data.root_pos_w[env_id]
                            - self.scene.env_origins[env_id]
                        ).tolist()
                    ],
                    "final_reference_local": [
                        float(value) for value in self._reference.position_w[env_id].tolist()
                    ],
                    "terminal_phase": int(self._supervisor.phase[env_id].item()),
                    "mission_success": bool(self._supervisor.mission_success[env_id].item()),
                    "mission_incomplete": bool(self._supervisor.mission_incomplete[env_id].item()),
                    "proposal_search_first_step": int(
                        self._proposal_search_first_step[env_id].item()
                    ),
                    "proposal_ledger_first_step": int(
                        self._proposal_ledger_first_step[env_id].item()
                    ),
                    "proposal_match_first_step": int(
                        self._proposal_match_first_step[env_id].item()
                    ),
                    "proposal_direct_reference": bool(
                        self._proposal_direct_reference[env_id].item()
                    ),
                    "proposal_count": self._supervisor.proposal_search.proposal_count,
                    "proposal_matched": int(
                        (
                            self._supervisor.proposal_search.matched_slot[env_id] >= 0
                        ).sum().item()
                    ),
                    "detector_backed_ledger": int(
                        ((self._supervisor.ledger.source[env_id] == 0) & self._supervisor.ledger.occupied[env_id])
                        .sum()
                        .item()
                    ),
                    "unresolved_proposals": int(
                        self._supervisor.proposal_search.unresolved_count[env_id].item()
                    ),
                    "phase_steps": [int(v) for v in phase_steps[row].tolist()],
                    "inspected": bool(inspected[env_id].item()),
                    "unique_openings_inspected": int(inspected_count[env_id].item()),
                    "validation_samples": int(self._supervisor.validation_samples_total[env_id].item()),
                    "validation_success": int(self._supervisor.validation_success_total[env_id].item()),
                    "validation_rejected": int(self._supervisor.validation_rejected_total[env_id].item()),
                    "validation_tof_invalid": int(self._supervisor.validation_tof_invalid_total[env_id].item()),
                    "validation_target_projected": int(self._validation_target_projected[env_id].item()),
                    "validation_opening_measured": int(self._validation_opening_measured[env_id].item()),
                    "validation_box_available": int(self._validation_box_available[env_id].item()),
                    "validation_tof_short": int(self._supervisor.validation_tof_short_total[env_id].item()),
                    "validation_clearance_low": int(
                        self._supervisor.validation_clearance_low_total[env_id].item()
                    ),
                    "validation_aim_off": int(self._supervisor.validation_aim_off_total[env_id].item()),
                    "validation_clearance_unknown": int(
                        self._supervisor.validation_clearance_unknown_total[env_id].item()
                    ),
                    "validation_accepted": int(self._supervisor.validation_accepted_total[env_id].item()),
                    "entry_events": int(self._entry_events[env_id].item()),
                    "repeat_entries": int(self._repeat_entries[env_id].item()),
                    "survey_steps": int(survey_steps[row].item()),
                    "survey_station_steps": int(self._survey_station_steps[env_id].item()),
                    "survey_travel_steps": int(self._survey_travel_steps[env_id].item()),
                    "survey_to_entry_events": int(self._survey_to_entry_events[env_id].item()),
                    "survey_to_entry_mean_s": (
                        float(self._survey_to_entry_sum_s[env_id].item())
                        / int(self._survey_to_entry_events[env_id].item())
                        if int(self._survey_to_entry_events[env_id].item()) > 0
                        else float("nan")
                    ),
                    "survey_to_entry_max_s": float(self._survey_to_entry_max_s[env_id].item()),
                    "yaw_rate_mean_abs": (
                        float(self._yaw_rate_abs_sum[env_id].item())
                        / int(self.episode_length_buf[env_id].item())
                        if int(self.episode_length_buf[env_id].item()) > 0
                        else float("nan")
                    ),
                    "yaw_rate_max_abs": float(self._yaw_rate_abs_max[env_id].item()),
                    "yaw_reversals": int(self._yaw_reversals[env_id].item()),
                    "observe_yaw_rate_mean_abs": (
                        float(self._observe_yaw_abs_sum[env_id].item())
                        / int(observe_steps[row].item())
                        if int(observe_steps[row].item()) > 0
                        else float("nan")
                    ),
                    "observe_yaw_rate_max_abs": float(self._observe_yaw_abs_max[env_id].item()),
                    "observe_yaw_reversals": int(self._observe_yaw_reversals[env_id].item()),
                }
            )

    def tof_grid(self) -> torch.Tensor:
        """Return the sparse ToF returns pooled to the actor grid."""

        self._refresh()
        grid = self._tof_points.image()[:, None]
        pooled = torch.nn.functional.adaptive_max_pool2d(-grid, self.cfg.tof_grid_shape)
        return (-pooled).flatten(1) / self.cfg.tof.max_range_m

    def _reset_idx(self, env_ids) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        phase_steps = self._phase_steps[ids]
        phase_total = phase_steps.sum(dim=1).clamp_min(1)
        timeout = self.termination_manager.get_term("time_out")[ids]
        collision = self.termination_manager.get_term("collision")[ids]
        recovery_failed = self.termination_manager.get_term("recovery_failed")[ids]
        end_phase = self._supervisor.phase[ids]
        ledger = self._supervisor.ledger
        action = self.action_manager.get_term("px4_waypoint")
        reward_weights = {
            name: self.reward_manager.get_term_cfg(name).weight
            for name in self.reward_manager.active_terms
        }
        reward_raw, reward_weighted = split_reward_totals(
            self.reward_manager._episode_sums, reward_weights, ids
        )
        minimum_distance = self._min_reference_distance[ids]
        minimum_distance = torch.where(
            torch.isfinite(minimum_distance), minimum_distance, torch.zeros_like(minimum_distance)
        )
        occupied = ledger.occupied[ids]
        solved = ledger.solved[ids] & occupied
        true_solved = self.true_solved_openings[ids]
        occupied_count = occupied.sum(dim=1)
        solved_count = solved.sum(dim=1)
        safe_occupied_count = occupied_count.clamp_min(1)
        safe_solved_count = solved_count.clamp_min(1)
        samples = self._supervisor.validation_samples_total[ids].float()
        tof_invalid = self._supervisor.validation_tof_invalid_total[ids].float()
        tof_blocked = self._supervisor.validation_tof_short_total[ids].float()
        def pooled(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
            total = denominator.sum()
            if total <= 0.0:
                return torch.full((), float("nan"), device=numerator.device)
            return numerator.sum() / total

        validation_steps = self._validation_steps[ids].float()
        facade_angle_error, facade_offset_error, on_briefed_face = self.facade_seed_error
        keep = on_briefed_face[ids]
        comparable = keep.float()
        zero = torch.zeros((), device=self.device)
        facade_angle_error = torch.where(keep, facade_angle_error[ids], zero)
        facade_offset_error = torch.where(keep, facade_offset_error[ids], zero)
        curriculum_samples = int(getattr(self, "_stage_episode_count", 0))
        if curriculum_samples > 0:
            curriculum_slice = slice(0, curriculum_samples)
            curriculum_success_rate = self._stage_successes[curriculum_slice].mean()
            curriculum_failure_rate = self._stage_failures[curriculum_slice].mean()
        else:
            curriculum_success_rate = torch.tensor(0.0, device=self.device)
            curriculum_failure_rate = torch.tensor(0.0, device=self.device)

        diagnostics = {
            "Mission/transit_arrived": self._transit_arrived[ids].float().mean(),
            "Mission/min_reference_distance_m": minimum_distance.mean(),
            "Mission/final_reference_distance_m": self._final_reference_distance[ids].mean(),
            "Mission/stations_scanned": self._supervisor.stations_scanned[ids].float().mean(),
            "Mission/success": self._supervisor.mission_success[ids].float().mean(),
            "Mission/incomplete": self._supervisor.mission_incomplete[ids].float().mean(),
            "Mission/proposals_unresolved": (
                self._supervisor.proposal_search.unresolved_count[ids].float().mean()
            ),
            "Mission/ledger_occupied": occupied_count.float().mean(),
            "Mission/ledger_solved": solved_count.float().mean(),
            "Mission/ledger_unsolved": (occupied_count - solved_count).float().mean(),
            "Mission/ledger_false_solved": (solved_count - true_solved.sum(dim=1)).clamp_min(0).float().mean(),
            "Mission/ledger_true_solved": true_solved.sum(dim=1).float().mean(),
            "Mission/ledger_true_validated": (self.true_validated_openings[ids].sum(dim=1).float().mean()),
            "Mission/ledger_true_inspected": (self.true_inspected_openings[ids].sum(dim=1).float().mean()),
            "Mission/ledger_terminal": ledger.terminal[ids].sum(dim=1).float().mean(),
            "Mission/candidate_confidence_mean": (
                (ledger.confidence[ids] * occupied).sum(dim=1) / safe_occupied_count
            ).mean(),
            "Mission/candidate_hits_mean": (
                (ledger.hits[ids] * occupied).sum(dim=1).float() / safe_occupied_count
            ).mean(),
            "Mission/candidate_conditioning_mean": (
                (ledger.conditioning[ids] * solved).sum(dim=1) / safe_solved_count
            ).mean(),
            "Perception/detector_proposals": self._detector_proposals[ids].float().mean(),
            "Perception/ledger_stale_pruned": ledger.stale_pruned_total[ids].float().mean(),
            "Perception/ledger_replaced": ledger.replaced_total[ids].float().mean(),
            "Perception/ledger_off_facade_pruned": (ledger.off_facade_pruned_total[ids].float().mean()),
            "Validation/samples": self._supervisor.validation_samples_total[ids].float().mean(),
            "Validation/tof_invalid": (self._supervisor.validation_tof_invalid_total[ids].float().mean()),
            "Validation/target_region_missing": (self._validation_target_region_missing[ids].float().mean()),
            "Validation/tof_short": (self._supervisor.validation_tof_short_total[ids].float().mean()),
            "Validation/clearance_low": (self._supervisor.validation_clearance_low_total[ids].float().mean()),
            "Validation/aim_off": (self._supervisor.validation_aim_off_total[ids].float().mean()),
            "Validation/clearance_unknown": (
                self._supervisor.validation_clearance_unknown_total[ids].float().mean()
            ),
            "Validation/success": self._supervisor.validation_success_total[ids].float().mean(),
            "Validation/evidence_valid_rate": pooled(samples - tof_invalid, samples),
            "Validation/blocked_rate": pooled(tof_blocked, samples),
            "Validation/clear_rate": pooled(
                self._supervisor.validation_accepted_total[ids].float(), samples
            ),
            "Mission/facade_hint_supplied": self._supervisor.facade_hint_known[ids].float().mean(),
            "Selection/facade_normal_error_deg": pooled(facade_angle_error, comparable),
            "Selection/facade_offset_error_m": pooled(facade_offset_error, comparable),
            "Selection/on_briefed_face_rate": comparable.mean(),
            "Selection/selected_truth_gap_m": (
                pooled(self._selected_truth_gap[ids], self._selected_gap_steps[ids].float())
            ),
            "Selection/true_occupied_rate": (
                pooled(self.true_occupied_openings[ids].sum(dim=1).float(), occupied_count.float())
            ),
            "Selection/true_confidence_mean": (
                pooled(self._selected_true_confidence[ids], self._selected_true_steps[ids].float())
            ),
            "Selection/false_confidence_mean": (
                pooled(self._selected_false_confidence[ids], self._selected_false_steps[ids].float())
            ),
            "Selection/true_hits_mean": (
                pooled(self._selected_true_hits[ids], self._selected_true_steps[ids].float())
            ),
            "Selection/false_hits_mean": (
                pooled(self._selected_false_hits[ids], self._selected_false_steps[ids].float())
            ),
            "Association/refresh_attempted": self._refresh_attempted[ids].float().mean(),
            "Association/refresh_found": self._refresh_found[ids].float().mean(),
            "Association/refresh_no_candidate": self._refresh_no_candidate[ids].float().mean(),
            "Association/refresh_ambiguous": self._refresh_ambiguous[ids].float().mean(),
            "Validation/tracks_any_rate": pooled(self._validation_tracks_any[ids].float(), validation_steps),
            "Validation/tracks_confirmed_rate": (
                pooled(self._validation_tracks_confirmed[ids].float(), validation_steps)
            ),
            "Validation/tracks_fresh_rate": (
                pooled(self._validation_tracks_fresh[ids].float(), validation_steps)
            ),
            "Association/stored_box_zero": self._assoc_stored_box_zero[ids].float().mean(),
            "Association/stored_box_gap": (
                pooled(self._assoc_box_gap_sum[ids], self._assoc_box_gap_count[ids].float())
            ),
            "Association/stored_id_absent": self._assoc_stored_id_absent[ids].float().mean(),
            "Association/stored_id_stale": self._assoc_stored_id_stale[ids].float().mean(),
            "Association/no_tracks": self._assoc_no_tracks[ids].float().mean(),
            "Association/no_identity": self._assoc_no_identity[ids].float().mean(),
            "Association/recovered": self._assoc_recovered[ids].float().mean(),
            "Association/by_identity": self._assoc_by_identity[ids].float().mean(),
            "Validation/target_box_available_rate": (
                pooled(self._validation_box_available[ids].float(), validation_steps)
            ),
            "Mission/selected_position_truth_match_rate": (
                pooled(self._validation_identity_correct[ids].float(), validation_steps)
            ),
            "Safety/collision": collision.float().mean(),
            "Safety/recovery_failed": recovery_failed.float().mean(),
            "Safety/unauthorized_crossing": (self._unauthorized_crossing_steps[ids] > 0).float().mean(),
            "Safety/inside_unauthorized_steps": self._inside_unauthorized_steps[ids].float().mean(),
            "Safety/crossing_observable_rate": pooled(
                self._crossing_observable_steps[ids].float(), self._crossing_scored_steps[ids].float()
            ),
            "Safety/facade_graze_steps": self._facade_graze_steps[ids].float().mean(),
            "Curriculum/window_success_rate": curriculum_success_rate,
            "Curriculum/window_failure_rate": curriculum_failure_rate,
            "Curriculum/window_samples": torch.tensor(float(curriculum_samples), device=self.device),
        }
        for name in self.reward_manager.active_terms:
            diagnostics[f"Reward_raw/{name}"] = reward_raw[name].mean()
            diagnostics[f"Reward_weighted/{name}"] = reward_weighted[name].mean()

        if self.cfg.entry_training.enabled:
            crossed_selected, crossed_elsewhere = self.entry_crossing
            entry_steps = self._entry_steps[ids].float()
            diagnostics.update(
                {
                    "Entry/success_rate": crossed_selected[ids].float().mean(),
                    "Entry/collision_rate": collision.float().mean(),
                    "Entry/unauthorized_crossing_rate": (
                        (self._unauthorized_crossing_steps[ids] > 0).float().mean()
                    ),
                    "Entry/wrong_opening_crossing_rate": crossed_elsewhere[ids].float().mean(),
                    "Entry/tof_veto_count": tof_blocked.mean(),
                    "Entry/crossing_time_s": (
                        (self.episode_length_buf[ids] * self.step_dt * crossed_selected[ids]).sum()
                        / crossed_selected[ids].sum().clamp_min(1)
                    ),
                    "Entry/closest_clearance_m": self._entry_closest_clearance_m[ids].mean(),
                    "Entry/target_visible_rate": pooled(self._entry_visible_steps[ids].float(), entry_steps),
                    "Entry/target_reacquired_rate": pooled(self._entry_reacquired[ids].float(), entry_steps),
                }
            )

        for phase in MissionPhase:
            diagnostics[f"Mission/phase_fraction_{phase.name}"] = (
                phase_steps[:, int(phase)].float() / phase_total
            ).mean()
            diagnostics[f"Mission/timeout_phase_{phase.name}"] = (timeout & (end_phase == int(phase))).float().mean()

        if self.cfg.record_episode_results:
            self._record_episode_results(
                ids, phase_steps, timeout, collision, recovery_failed, action, reward_raw, reward_weighted
            )

        super()._reset_idx(env_ids)
        self.extras["log"].update(diagnostics)
        self._supervisor.reset(ids)
        if self.cfg.entry_training.enabled:
            normal = torch.tensor(
                self.cfg.entry_training.truth_facade_normal_w, device=self.device
            ).expand(ids.numel(), 3)
            standoff = ((self.entry_start_w[ids] - self.entry_target_w[ids]) * normal).sum(dim=-1)
            anchor = self.entry_start_w[ids] - normal * standoff[:, None]
            self._supervisor.begin_at_survey(ids, anchor, normal, station_standoff_m=standoff)
        self._trackers.reset(ids)
        self._disparity.reset(ids)
        self.clearance_m[ids] = self.cfg.max_clearance_m
        self.clearance_known[ids] = False
        self._phase_steps[ids] = 0
        self._transit_arrived[ids] = False
        self._min_reference_distance[ids] = torch.inf
        self._final_reference_distance[ids] = 0.0
        self._detector_proposals[ids] = 0
        self._proposal_search_first_step[ids] = -1
        self._proposal_ledger_first_step[ids] = -1
        self._proposal_match_first_step[ids] = -1
        self._proposal_direct_reference[ids] = False
        self._validation_target_region_missing[ids] = 0
        self._validation_target_projected[ids] = 0
        self._validation_opening_measured[ids] = 0
        self._validation_steps[ids] = 0
        self._validation_box_available[ids] = 0
        self._validation_identity_correct[ids] = 0
        self._unauthorized_crossing_steps[ids] = 0
        self._crossing_scored_steps[ids] = 0
        self._crossing_observable_steps[ids] = 0
        self._inside_unauthorized_steps[ids] = 0
        self._was_behind_facade[ids] = False
        self._max_facade_depth_m[ids] = 0.0
        self._facade_graze_steps[ids] = 0
        self._crossed_inward_near[ids] = False
        self._crossed_inward_far[ids] = False
        self._observe_run_s[ids] = 0.0
        self._was_observing[ids] = False
        self._min_observe_hold_s[ids] = torch.inf
        self._unauthorized_phase[ids] = -1
        self._previous_phase[ids] = int(MissionPhase.TRANSIT)
        self._survey_station_steps[ids] = 0
        self._survey_travel_steps[ids] = 0
        self._since_survey_s[ids] = 0.0
        self._survey_to_entry_sum_s[ids] = 0.0
        self._survey_to_entry_max_s[ids] = 0.0
        self._survey_to_entry_events[ids] = 0
        self._entry_events[ids] = 0
        self._repeat_entries[ids] = 0
        self._entered_slot[ids] = False
        self._yaw_rate_abs_sum[ids] = 0.0
        self._yaw_rate_abs_max[ids] = 0.0
        self._yaw_reversals[ids] = 0
        self._previous_yaw_rate[ids] = 0.0
        self._previous_yaw_w[ids] = 0.0
        self._yaw_history_valid[ids] = False
        self._reward_previous_distance[ids] = 0.0
        self._reward_previous_reference_phase[ids] = int(MissionPhase.TRANSIT)
        self._reward_reference_history_valid[ids] = False
        self._reward_previous_box_error[ids] = 0.0
        self._reward_previous_box_phase[ids] = int(MissionPhase.TRANSIT)
        self._reward_box_history_valid[ids] = False
        self._reward_previous_yaw_error[ids] = 0.0
        self._reward_previous_yaw_phase[ids] = int(MissionPhase.TRANSIT)
        self._reward_yaw_progress_valid[ids] = False
        self._reward_previous_yaw_rate[ids] = 0.0
        self._reward_yaw_history_valid[ids] = False
        self._previous_transit_completion[ids] = 0.0
        self._previous_station[ids] = 0.0
        self._previous_observing[ids] = 0.0
        self._previous_consumed[ids] = 0.0
        self._previous_inspected[ids] = 0.0
        self._previous_safe_exit[ids] = 0.0
        self._previous_floor[ids] = 0.0
        self._previous_entry_crossing[ids] = 0.0
        self._previous_wrong_opening[ids] = 0.0
        self._observe_yaw_abs_sum[ids] = 0.0
        self._observe_yaw_abs_max[ids] = 0.0
        self._observe_yaw_reversals[ids] = 0
        self._in_validation[ids] = False
        self._entry_steps[ids] = 0
        self._entry_visible_steps[ids] = 0
        self._entry_reacquired[ids] = 0
        self._entry_closest_clearance_m[ids] = self.cfg.max_clearance_m
        self._entry_crossing_depth_m[ids] = 0.0
        self.unauthorized_crossing_now[ids] = False
        self._selected_truth_gap[ids] = 0.0
        self._selected_gap_steps[ids] = 0
        self._selected_true_steps[ids] = 0
        self._selected_false_steps[ids] = 0
        self._selected_true_confidence[ids] = 0.0
        self._selected_false_confidence[ids] = 0.0
        self._selected_true_hits[ids] = 0.0
        self._selected_false_hits[ids] = 0.0
        self._refresh_attempted[ids] = 0
        self._refresh_found[ids] = 0
        self._refresh_no_candidate[ids] = 0
        self._refresh_ambiguous[ids] = 0
        self._validation_tracks_any[ids] = 0
        self._validation_tracks_confirmed[ids] = 0
        self._validation_tracks_fresh[ids] = 0
        self._assoc_stored_box_zero[ids] = 0
        self._assoc_box_gap_sum[ids] = 0.0
        self._assoc_box_gap_count[ids] = 0
        self._assoc_stored_id_absent[ids] = 0
        self._assoc_stored_id_stale[ids] = 0
        self._assoc_no_tracks[ids] = 0
        self._assoc_no_identity[ids] = 0
        self._assoc_recovered[ids] = 0
        self._assoc_by_identity[ids] = 0
        self.active_opening_box[ids] = 0
        self.active_opening_box_valid[ids] = 0
        self.active_opening_box_centered[ids] = 0
        self.active_opening_confidence[ids] = 0
        self.active_opening_age_s[ids] = 0
        self.active_target_changed[ids] = 0
        self._previous_active_slot[ids] = -1
        self.tof_point_count[ids] = 0
        self.tof_threshold_configured[ids] = False
        self._mission_step = -1
