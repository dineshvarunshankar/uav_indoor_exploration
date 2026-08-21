"""Custom environment owning casualty-inspection state and perception."""

from __future__ import annotations

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat

from .crowd.animation import AnimatedCrowdAdapter
from .crowd.manager import Behavior, CrowdManager
from .mission import (
    FINAL_STAGE,
    STAGE_HAZARDS_ACTIVE,
    STAGE_TARGET_COUNT,
    STAGE_TERMINAL_PHASE,
    InspectionMissionSupervisor,
    MissionPhase,
    MissionReference,
    MissionSupervisorCfg,
    minimum_scene_clearance,
    point_navigation_reference,
)
from .perception.pipeline import InspectionPerception, InspectionPerceptionCfg


class CasualtyInspectionEnv(ManagerBasedRLEnv):
    """Model-in-loop local navigation task with a deterministic mission supervisor."""

    cfg: object

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        device = torch.device(cfg.sim.device)
        num_envs = cfg.scene.num_envs
        max_casualties = cfg.max_casualties
        max_hazards = cfg.max_hazards

        perception_cfg = InspectionPerceptionCfg(
            midas_model_path=cfg.midas_model_path,
            pose_model_path=cfg.pose_model_path,
            pose_device=cfg.pose_device,
            photometric_randomization=cfg.photometric_randomization,
            frame_drop_probability=cfg.frame_drop_probability,
        )
        self._perception = InspectionPerception(num_envs, device, perception_cfg)
        self._supervisor = InspectionMissionSupervisor(
            num_envs,
            max_casualties,
            device,
            MissionSupervisorCfg(),
        )

        self.casualty_truth_w = torch.zeros(num_envs, max_casualties, 3, device=device)
        self.casualty_active = torch.zeros(num_envs, max_casualties, dtype=torch.bool, device=device)
        self.dynamic_hazard_positions_w = torch.zeros(num_envs, max_hazards, 3, device=device)
        self.hazard_active = torch.zeros(num_envs, max_hazards, dtype=torch.bool, device=device)
        self.gps_targets_w = torch.zeros_like(self.casualty_truth_w)
        self.target_counts = torch.ones(num_envs, dtype=torch.long, device=device)
        self._minimum_clearance = torch.full((num_envs,), 50.0, device=device)
        self._recording_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._arc_frame_lower_quantile = torch.zeros(num_envs, device=device)
        self._arc_frame_samples = torch.zeros(num_envs, 100, device=device)
        self._arc_frame_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._previous_mission_phase = torch.full(
            (num_envs,),
            int(MissionPhase.APPROACH),
            dtype=torch.long,
            device=device,
        )
        self._wrong_person = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._hazard_center = torch.zeros_like(self.dynamic_hazard_positions_w)
        self._hazard_velocity = torch.zeros_like(self.dynamic_hazard_positions_w)
        self._hazard_phase = torch.zeros(num_envs, max_hazards, device=device)
        self._episode_start_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._mission_progress = torch.zeros(num_envs, device=device)
        self._mission_progress_delta = torch.zeros(num_envs, device=device)
        self._mission_step = 0
        self._outcome_history = torch.zeros(512, device=device)
        self._outcome_index = 0
        self._outcome_filled = 0
        self.stage_started_step = 0
        self._mission_reference = MissionReference(
            position_w=torch.zeros(num_envs, 3, device=device),
            yaw_w=torch.zeros(num_envs, device=device),
            phase=torch.full((num_envs,), int(MissionPhase.APPROACH), dtype=torch.long, device=device),
            target_valid=torch.zeros(num_envs, dtype=torch.bool, device=device),
            recording=torch.zeros(num_envs, dtype=torch.bool, device=device),
            arc_progress=torch.zeros(num_envs, device=device),
        )

        super().__init__(cfg, render_mode, **kwargs)
        self.crowd = CrowdManager(
            num_envs=self.num_envs,
            device=str(self.device),
            env_origins=self.scene.env_origins,
        )
        self.crowd_animation = AnimatedCrowdAdapter(
            num_envs=self.num_envs,
            num_people=self.crowd.num_people,
            num_casualties=self.crowd.num_targets,
            device=self.device,
        )
        self.crowd_animation.attach()
        # Casualties are static, so their pose is written once. Posture and facing vary per
        # casualty per environment, otherwise the drone only ever learns one approach.
        casualty_position = self.crowd.position[:, : self.crowd.num_targets].clone()
        casualty_position[..., 2] += self.crowd.ground_clearance_m[:, : self.crowd.num_targets]
        self.crowd_animation.place_casualties(
            casualty_position,
            self.crowd.orientation_wxyz()[:, : self.crowd.num_targets],
            self.scene.env_origins,
        )

    def _ensure_perception_current(self) -> None:
        """Run perception and the mission once per env step, on the post-physics state.

        ``common_step_counter`` increments straight after the physics loop and before
        terminations, so the first access from any manager term refreshes the state the
        policy is judged against. Mirrors how Isaac Lab sensors refresh their buffers.
        """

        if self._mission_step == self.common_step_counter:
            return
        self._mission_step = self.common_step_counter
        self._update_models_and_mission()

    @property
    def perception(self):
        self._ensure_perception_current()
        return self._perception

    @property
    def supervisor(self):
        self._ensure_perception_current()
        return self._supervisor

    @property
    def mission_reference(self) -> MissionReference:
        self._ensure_perception_current()
        return self._mission_reference

    @property
    def minimum_clearance(self) -> torch.Tensor:
        self._ensure_perception_current()
        return self._minimum_clearance

    @property
    def recording_valid(self) -> torch.Tensor:
        self._ensure_perception_current()
        return self._recording_valid

    @property
    def arc_frame_lower_quantile(self) -> torch.Tensor:
        self._ensure_perception_current()
        return self._arc_frame_lower_quantile

    @property
    def wrong_person(self) -> torch.Tensor:
        self._ensure_perception_current()
        return self._wrong_person

    @property
    def mission_progress_delta(self) -> torch.Tensor:
        self._ensure_perception_current()
        return self._mission_progress_delta

    @property
    def episode_success_rate(self) -> float:
        if self._outcome_filled == 0:
            return 0.0
        return float(self._outcome_history[: self._outcome_filled].mean())

    def on_curriculum_promoted(self) -> None:
        """Start the next stage with a clean success window."""

        self.stage_started_step = self.common_step_counter
        self._outcome_history.zero_()
        self._outcome_index = 0
        self._outcome_filled = 0

    def _record_episode_outcome(self, completed: torch.Tensor) -> None:
        capacity = self._outcome_history.numel()
        slots = (torch.arange(completed.numel(), device=self.device) + self._outcome_index) % capacity
        self._outcome_history[slots] = completed.float()
        self._outcome_index = int((self._outcome_index + completed.numel()) % capacity)
        self._outcome_filled = min(self._outcome_filled + completed.numel(), capacity)

    def _reset_idx(self, env_ids) -> None:
        completed = (self._mission_reference.phase[env_ids] == int(MissionPhase.DONE)) & (
            ~self._supervisor.failed[env_ids]
        )
        self._record_episode_outcome(completed)
        super()._reset_idx(env_ids)

    def _reference_for_point_navigation(
        self,
        robot_position: torch.Tensor,
        robot_yaw: torch.Tensor,
    ) -> MissionReference:
        env_index = torch.arange(self.num_envs, device=self.device)
        goal = self.gps_targets_w[env_index, torch.zeros(self.num_envs, dtype=torch.long, device=self.device)]
        return point_navigation_reference(robot_position, robot_yaw, goal)

    def _update_models_and_mission(self) -> None:
        camera = self.scene["rgb_camera"]
        rgb = camera.data.output["rgb"]
        self._perception.tick(self.step_dt, rgb)

        robot = self.scene["robot"]
        _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        yaw = torch.remainder(yaw + math.pi, 2.0 * math.pi) - math.pi
        track_positions = self._perception.estimated_track_positions_w(robot.data.root_pos_w, yaw)
        tracker = self._perception.tracker

        if self.cfg.curriculum_stage == 0:
            self._mission_reference = self._reference_for_point_navigation(robot.data.root_pos_w, yaw)
        else:
            self._mission_reference = self._supervisor.update(
                self.step_dt,
                robot.data.root_pos_w,
                yaw,
                self.gps_targets_w,
                self.target_counts,
                track_positions,
                tracker.confidence,
                tracker.age_s,
                tracker.face_confidence >= 0.35,
                tracker.state[..., 1],
            )

        self._update_privileged_metrics(track_positions)

    def _update_privileged_metrics(self, estimated_tracks_w: torch.Tensor) -> None:
        robot_position = self.scene["robot"].data.root_pos_w
        self._minimum_clearance = minimum_scene_clearance(
            robot_position, self.dynamic_hazard_positions_w, self.hazard_active, self.crowd.position
        )

        selected = self._supervisor.selected_track.clamp_min(0)
        env_index = torch.arange(self.num_envs, device=self.device)
        selected_position = estimated_tracks_w[env_index, selected]
        closest_truth = torch.linalg.vector_norm(
            self.casualty_truth_w - selected_position[:, None, :],
            dim=-1,
        ).argmin(dim=1)
        self._wrong_person = (self._supervisor.selected_track >= 0) & (
            closest_truth != self._supervisor.target_index.clamp_max(self.cfg.max_casualties - 1)
        )

        selected_face = self._perception.tracker.face_confidence[env_index, selected]
        position_error = torch.linalg.vector_norm(
            self._mission_reference.position_w - robot_position,
            dim=-1,
        )
        self._recording_valid = (
            self._mission_reference.recording
            & (selected_face >= 0.35)
            & (position_error <= self._supervisor.cfg.record_position_tolerance_m)
        )
        entered_recording = self._mission_reference.recording & (
            self._previous_mission_phase != int(MissionPhase.RECORD)
        )
        self._arc_frame_samples[entered_recording] = 0.0
        self._arc_frame_count[entered_recording] = 0
        recording_envs = torch.nonzero(self._mission_reference.recording, as_tuple=False).flatten()
        if recording_envs.numel() > 0:
            sample_index = self._arc_frame_count[recording_envs].clamp_max(self._arc_frame_samples.shape[1] - 1)
            self._arc_frame_samples[recording_envs, sample_index] = selected_face[recording_envs]
            self._arc_frame_count[recording_envs] = (self._arc_frame_count[recording_envs] + 1).clamp_max(
                self._arc_frame_samples.shape[1]
            )
            sample_slots = torch.arange(self._arc_frame_samples.shape[1], device=self.device)[None, :]
            valid_samples = sample_slots < self._arc_frame_count[:, None]
            sortable = torch.where(
                valid_samples, self._arc_frame_samples, torch.full_like(self._arc_frame_samples, torch.inf)
            )
            sorted_samples = torch.sort(sortable, dim=1).values
            quantile_index = torch.floor(0.1 * (self._arc_frame_count - 1).clamp_min(0)).long()
            env_index = torch.arange(self.num_envs, device=self.device)
            lower_quantile = sorted_samples[env_index, quantile_index]
            self._arc_frame_lower_quantile = torch.where(
                self._arc_frame_count > 0,
                lower_quantile,
                torch.zeros_like(lower_quantile),
            )
        progress = self._supervisor.completed_targets.float() * 5.0 + (
            self._mission_reference.phase.clamp_max(int(MissionPhase.RECORD)).float()
        )
        self._mission_progress_delta = (progress - self._mission_progress).clamp_min(0.0)
        self._mission_progress = progress
        self._previous_mission_phase[:] = self._mission_reference.phase

    def _write_crowd_state(self) -> None:
        # The capsule must follow the posture. A lying casualty whose collision body stayed
        # upright would be an invisible 1.7 m pole the drone cannot see and cannot descend
        # past, and minimum_clearance would measure the pole instead of the person.
        proxy_position = self.crowd.position.clone()
        proxy_position[..., 2] += self.crowd.body_centre_height_m
        proxy_pose = torch.cat((proxy_position, self.crowd.orientation_wxyz()), dim=-1)
        self.scene["person_proxies"].write_object_pose_to_sim(proxy_pose)
        self.casualty_truth_w[:] = self.crowd.position[:, : self.cfg.max_casualties]

    def _update_crowd(self) -> None:
        self.crowd.step(self.step_dt, self.cfg.curriculum_stage)
        moving = (self.crowd.behavior == int(Behavior.PATROL)) | (
            self.crowd.behavior == int(Behavior.WANDER)
        )
        self.crowd_animation.command(
            self.crowd.destination,
            self.crowd.speed,
            moving,
        )
        self._write_crowd_state()

    def _update_dynamic_hazards(self) -> None:
        elapsed = (self.common_step_counter - self._episode_start_step).float() * self.step_dt
        phase = elapsed[:, None] + self._hazard_phase
        position = self._hazard_center + self._hazard_velocity * torch.sin(phase)[..., None]
        position[:, 1, 2] += 0.7 * torch.sin(0.7 * phase[:, 1])
        inactive_position = self.scene.env_origins[:, None, :] + torch.tensor(
            [0.0, 0.0, -20.0],
            device=self.device,
        )
        position = torch.where(self.hazard_active[..., None], position, inactive_position)
        self.dynamic_hazard_positions_w[:] = position

        identity = torch.zeros(self.num_envs, 4, device=self.device)
        identity[:, 0] = 1.0
        self.scene["dynamic_vehicle"].write_root_pose_to_sim(torch.cat((position[:, 0], identity), dim=-1))
        self.scene["dynamic_drone"].write_root_pose_to_sim(torch.cat((position[:, 1], identity), dim=-1))

    def step(self, action):
        self._update_crowd()
        self._update_dynamic_hazards()
        observation, reward, terminated, truncated, extras = super().step(action)
        self.crowd.position[:] = self.crowd_animation.positions()
        self._write_crowd_state()
        extras.setdefault("log", {}).update(self._mission_metrics())
        return observation, reward, terminated, truncated, extras

    def reset_episode(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        self.crowd.on_drone_reset(env_ids)
        self.casualty_truth_w[env_ids] = self.crowd.position[env_ids, : self.cfg.max_casualties]
        stage = min(self.cfg.curriculum_stage, FINAL_STAGE)
        self._supervisor.terminal_phase = STAGE_TERMINAL_PHASE[stage]
        target_count = STAGE_TARGET_COUNT[stage]
        self.target_counts[env_ids] = target_count
        indices = torch.arange(self.cfg.max_casualties, device=self.device)
        self.casualty_active[env_ids] = indices[None, :] < target_count

        noise_direction = torch.rand(count, self.cfg.max_casualties, device=self.device) * 2.0 * math.pi
        low, high = self.cfg.gps_uncertainty_m
        noise_radius = low + (high - low) * torch.rand(count, self.cfg.max_casualties, device=self.device)
        noise = torch.zeros_like(self.casualty_truth_w[env_ids])
        noise[..., 0] = noise_radius * torch.cos(noise_direction)
        noise[..., 1] = noise_radius * torch.sin(noise_direction)
        self.gps_targets_w[env_ids] = self.casualty_truth_w[env_ids] + noise

        self._write_crowd_state()

        self._hazard_center[env_ids] = origins[:, None, :] + torch.tensor(
            [[6.0, 0.0, 0.5], [-5.0, 0.0, 2.5]],
            device=self.device,
        )
        direction = torch.rand(count, self.cfg.max_hazards, device=self.device) * 2.0 * math.pi
        speed = 0.5 + 1.5 * torch.rand(count, self.cfg.max_hazards, device=self.device)
        self._hazard_velocity[env_ids, :, 0] = speed * torch.cos(direction)
        self._hazard_velocity[env_ids, :, 1] = speed * torch.sin(direction)
        self._hazard_velocity[env_ids, :, 2] = 0.0
        self._hazard_phase[env_ids] = torch.rand(count, self.cfg.max_hazards, device=self.device) * 2.0 * math.pi
        active_hazards = self.cfg.max_hazards if STAGE_HAZARDS_ACTIVE[stage] else 0
        self.hazard_active[env_ids] = torch.arange(self.cfg.max_hazards, device=self.device)[None, :] < active_hazards
        self._episode_start_step[env_ids] = self.common_step_counter

        self._supervisor.reset(env_ids)
        self._perception.reset(env_ids)
        self._arc_frame_lower_quantile[env_ids] = 0.0
        self._arc_frame_samples[env_ids] = 0.0
        self._arc_frame_count[env_ids] = 0
        self._previous_mission_phase[env_ids] = int(MissionPhase.APPROACH)
        self._mission_progress[env_ids] = 0.0
        self._mission_progress_delta[env_ids] = 0.0
        self._recording_valid[env_ids] = False
        self._wrong_person[env_ids] = False

        robot = self.scene["robot"]
        _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        goal = self.gps_targets_w[:, 0]
        self._mission_reference.position_w[env_ids] = goal[env_ids]
        self._mission_reference.yaw_w[env_ids] = yaw[env_ids]
        self._mission_reference.phase[env_ids] = self._supervisor.phase[env_ids]
        self._mission_reference.target_valid[env_ids] = False
        self._mission_reference.recording[env_ids] = False
        self._mission_reference.arc_progress[env_ids] = 0.0
        if not hasattr(self, "_previous_reference_distance"):
            self._previous_reference_distance = torch.zeros(self.num_envs, device=self.device)
            self._previous_reference_phase = self._supervisor.phase.clone()
        self._previous_reference_distance[env_ids] = torch.linalg.vector_norm(
            goal[env_ids] - robot.data.root_pos_w[env_ids],
            dim=-1,
        )
        self._previous_reference_phase[env_ids] = self._supervisor.phase[env_ids]

    def _mission_metrics(self) -> dict[str, torch.Tensor]:
        action = self.action_manager.get_term("px4_velocity")
        phase = self._mission_reference.phase
        return {
            "CasualtyInspection/mission_completion": (
                (phase == int(MissionPhase.DONE)) & ~self._supervisor.failed
            ).float().mean(),
            "CasualtyInspection/mission_failure": self._supervisor.failed.float().mean(),
            "CasualtyInspection/wrong_person": self._wrong_person.float().mean(),
            "CasualtyInspection/face_acquired": ((self._perception.tracker.face_confidence >= 0.35) & self._perception.tracker.valid)
            .any(dim=1)
            .float()
            .mean(),
            "CasualtyInspection/valid_recording": self._recording_valid.float().mean(),
            "CasualtyInspection/arc_frame_lower_quantile": self._arc_frame_lower_quantile.mean(),
            "CasualtyInspection/minimum_clearance_m": self._minimum_clearance.mean(),
            "CasualtyInspection/shield_intervention": action.shield_intervened.float().mean(),
            "CasualtyInspection/stale_depth": (self._perception.depth_age_s > 0.2).float().mean(),
            "CasualtyInspection/stale_pose": (self._perception.pose_age_s > 0.2).float().mean(),
            "Latency/midas_ms": torch.tensor(self._perception.last_depth_latency_s * 1000.0, device=self.device),
            "Latency/pose_ms": torch.tensor(self._perception.last_pose_latency_s * 1000.0, device=self.device),
        }
