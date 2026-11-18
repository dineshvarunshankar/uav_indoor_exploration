"""Standalone random-goal obstacle-navigation environment."""

from __future__ import annotations

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from ..building_inspection.reward_accounting import split_reward_totals
from ..inspection import MissionPhase, MissionReference
from ..perception import DisparityMonitor
from .curriculum import CurriculumState, sample_difficulties
from .geometry import nearest_geometry
from .layouts import (
    LayoutDifficulty,
    LazyLayoutBank,
    NavigationLayout,
    generate_layout,
    layout_bank_index,
    obstacle_slot_specs,
)
from .policy import clipped_continuous_fraction
from .reward_features import (
    blended_yaw_direction,
    physical_clearance,
    successful_episode,
)


class ObstacleNavigationEnv(ManagerBasedRLEnv):
    """Use deployed disparity to reach one hidden-feasible random 3D goal."""

    cfg: object

    def __init__(self, cfg, render_mode: str | None = None, **kwargs) -> None:
        device = torch.device(cfg.sim.device)
        count = cfg.scene.num_envs
        self.held_out_seed_start = cfg.held_out_seed_start
        self.layout_bank_seed_start = (
            cfg.held_out_seed_start if cfg.use_held_out_layouts else cfg.training_seed_start
        )
        self.training_layout_bank: dict[int, LazyLayoutBank[NavigationLayout]] = {
            difficulty: LazyLayoutBank(
                cfg.layout_bank_size,
                lambda index, difficulty=difficulty: generate_layout(
                    seed=self.layout_bank_seed_start + difficulty * cfg.layout_bank_size + index,
                    difficulty=LayoutDifficulty(difficulty),
                    vehicle_radius_m=cfg.vehicle_radius_m,
                    margin_m=cfg.training_clearance_margin_m,
                    endpoint_xy_boundary_buffer_m=cfg.endpoint_xy_boundary_buffer_m,
                    obstacle_count_override=cfg.obstacle_count_overrides[difficulty],
                ),
            )
            for difficulty in range(len(LayoutDifficulty))
        }
        self.curriculum_state = CurriculumState(
            current_difficulty=cfg.initial_difficulty,
            rehearsal_boosts=tuple(float(value) for value in cfg.rehearsal_boosts),
        )
        self._layout_reset_count = 0
        self._layout_episode_count = torch.zeros(count, dtype=torch.long, device=device)
        self._disparity = DisparityMonitor(count, device, cfg.disparity)
        self._disparity_step = -1

        self.goal_w = torch.zeros(count, 3, device=device)
        self.layout_seed = torch.zeros(count, dtype=torch.long, device=device)
        self.bounds_min_m = torch.zeros(count, 3, device=device)
        self.bounds_max_m = torch.zeros(count, 3, device=device)
        self.layout_difficulty = torch.zeros(count, dtype=torch.long, device=device)
        self.layout_route_family = ["unclassified"] * count
        self.layout_detour_direction = ["unclassified"] * count
        self.layout_narrow_passage = torch.zeros(count, dtype=torch.bool, device=device)
        self.layout_direct_path_blocked = torch.zeros(count, dtype=torch.bool, device=device)
        self.layout_minimum_route_clearance_m = torch.full((count,), torch.inf, device=device)
        slot_specs = obstacle_slot_specs()
        slots = len(slot_specs)
        self.obstacle_cylinder_slots = torch.tensor(
            [spec.family == "post" for spec in slot_specs], dtype=torch.bool, device=device
        )
        self.obstacle_centres_w = torch.zeros(count, slots, 3, device=device)
        self.obstacle_sizes_m = torch.zeros(count, slots, 3, device=device)
        self.obstacle_active = torch.zeros(count, slots, dtype=torch.bool, device=device)
        self.state_estimate_fresh = torch.ones(count, dtype=torch.bool, device=device)

        self._reward_previous_distance = torch.zeros(count, device=device)
        self._reward_distance_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self._reward_goal_latched = torch.zeros(count, dtype=torch.bool, device=device)
        self._reward_episode_safe = torch.ones(count, dtype=torch.bool, device=device)
        self._reward_previous_yaw_error = torch.zeros(count, device=device)
        self._reward_yaw_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self._reward_previous_yaw_rate = torch.zeros(count, device=device)
        self._reward_yaw_rate_valid = torch.zeros(count, dtype=torch.bool, device=device)
        self._reward_previous_acceleration_w = torch.zeros(count, 3, device=device)
        self._reward_acceleration_valid = torch.zeros(count, dtype=torch.bool, device=device)

        self.completed_records: list[dict[str, object]] = []
        self._metric_previous_position_w = torch.zeros(count, 3, device=device)
        self._metric_path_length_m = torch.zeros(count, device=device)
        self._metric_reference_length_m = torch.ones(count, device=device)
        self._metric_minimum_clearance_m = torch.full((count,), torch.inf, device=device)
        self._metric_altitude_violation_s = torch.zeros(count, device=device)
        self._metric_stall_time_s = torch.zeros(count, device=device)
        self._metric_body_rate_max_rad_s = torch.zeros(count, device=device)
        self._metric_previous_yaw_rate_rad_s = torch.zeros(count, device=device)
        self._metric_yaw_acceleration_max_rad_s2 = torch.zeros(count, device=device)

        phase = torch.full((count,), int(MissionPhase.TRANSIT), dtype=torch.long, device=device)
        self._mission_reference = MissionReference(
            position_w=self.goal_w,
            yaw_w=torch.zeros(count, device=device),
            phase=phase,
            entry_authorized=torch.zeros(count, dtype=torch.bool, device=device),
            crossing_opening=torch.zeros(count, dtype=torch.bool, device=device),
            target_valid=torch.ones(count, dtype=torch.bool, device=device),
            detector_enabled=torch.zeros(count, dtype=torch.bool, device=device),
            accepts_proposals=torch.zeros(count, dtype=torch.bool, device=device),
            observation_progress=torch.zeros(count, device=device),
        )
        super().__init__(cfg, render_mode, **kwargs)

    @property
    def disparity(self) -> DisparityMonitor:
        if self._disparity_step != self.common_step_counter:
            self._disparity_step = self.common_step_counter
            output = self.scene["rgb_camera"].data.output
            self._disparity.tick(self.step_dt, output["rgb"], output.get("distance_to_image_plane"))
        return self._disparity

    @property
    def mission_reference(self) -> MissionReference:
        robot = self.scene["robot"]
        position = robot.data.root_pos_w - self.scene.env_origins
        delta = self.goal_w - position
        horizontal_velocity = self.scene["robot"].data.root_lin_vel_w[:, :2]
        desired_heading = blended_yaw_direction(
            goal_direction_w=delta[:, :2],
            horizontal_velocity_w=horizontal_velocity,
            motion_speed_m_s=torch.linalg.vector_norm(horizontal_velocity, dim=-1),
            full_motion_weight_speed_m_s=self.cfg.yaw_motion_blend_speed_m_s,
        )
        self._mission_reference.position_w = self.goal_w
        self._mission_reference.yaw_w = torch.atan2(
            desired_heading[:, 1], desired_heading[:, 0]
        )
        return self._mission_reference

    @property
    def robot_yaw_w(self) -> torch.Tensor:
        _, _, yaw = euler_xyz_from_quat(self.scene["robot"].data.root_quat_w)
        return yaw

    def _select_layouts(self, ids: torch.Tensor) -> list[NavigationLayout]:
        if self.cfg.evaluation_difficulty is None:
            difficulties = sample_difficulties(
                self.curriculum_state,
                count=ids.numel(),
                seed=self.layout_bank_seed_start + self._layout_reset_count,
            )
        else:
            difficulties = [int(self.cfg.evaluation_difficulty)] * ids.numel()
        selected: list[NavigationLayout] = []
        for offset, difficulty in enumerate(difficulties):
            bank = self.training_layout_bank[difficulty]
            env_id = int(ids[offset])
            index = layout_bank_index(
                env_id, int(self._layout_episode_count[env_id].item()), len(bank)
            )
            selected.append(bank[index])
            self.layout_difficulty[ids[offset]] = difficulty
            self.layout_seed[ids[offset]] = (
                self.layout_bank_seed_start + difficulty * self.cfg.layout_bank_size + index
            )
            self.layout_route_family[env_id] = bank[index].route_family
            self.layout_detour_direction[env_id] = bank[index].detour_direction
            self.layout_narrow_passage[ids[offset]] = bank[index].narrow_passage
            self.layout_direct_path_blocked[ids[offset]] = bank[index].direct_path_blocked
            self.layout_minimum_route_clearance_m[ids[offset]] = (
                bank[index].minimum_route_clearance_m
            )
        self._layout_episode_count[ids] += 1
        self._layout_reset_count += ids.numel()
        return selected

    def _update_episode_metrics(self, ids: torch.Tensor) -> None:
        if ids.numel() == 0:
            return
        position_w = self.scene["robot"].data.root_pos_w[ids]
        position = position_w - self.scene.env_origins[ids]
        self._metric_path_length_m[ids] += torch.linalg.vector_norm(
            position_w - self._metric_previous_position_w[ids], dim=-1
        )
        self._metric_previous_position_w[ids] = position_w

        obstacle_distance, _, obstacle_known = nearest_geometry(
            position,
            self.obstacle_centres_w[ids],
            self.obstacle_sizes_m[ids],
            self.obstacle_active[ids],
            self.obstacle_cylinder_slots,
        )
        clearance = physical_clearance(
            obstacle_distance_m=obstacle_distance,
            obstacle_known=obstacle_known,
            height_above_ground_m=position[:, 2],
            vehicle_radius_m=self.cfg.vehicle_radius_m,
        )
        self._metric_minimum_clearance_m[ids] = torch.minimum(
            self._metric_minimum_clearance_m[ids], clearance
        )

        self._metric_altitude_violation_s[ids] += (
            position[:, 2] < self.cfg.minimum_altitude_m
        ).float() * self.step_dt
        speed = torch.linalg.vector_norm(self.scene["robot"].data.root_lin_vel_w[ids], dim=-1)
        goal_distance = torch.linalg.vector_norm(self.goal_w[ids] - position, dim=-1)
        stalled = (speed < 0.10) & (goal_distance > self.cfg.goal_tolerance_m)
        stalled &= self.episode_length_buf[ids] > 10
        self._metric_stall_time_s[ids] += stalled.float() * self.step_dt
        body_rate = torch.linalg.vector_norm(
            self.scene["robot"].data.root_ang_vel_b[ids], dim=-1
        )
        self._metric_body_rate_max_rad_s[ids] = torch.maximum(
            self._metric_body_rate_max_rad_s[ids], body_rate
        )
        yaw_rate = self.scene["robot"].data.root_ang_vel_b[ids, 2]
        yaw_acceleration = (
            (yaw_rate - self._metric_previous_yaw_rate_rad_s[ids]) / self.step_dt
        ).abs()
        self._metric_yaw_acceleration_max_rad_s2[ids] = torch.maximum(
            self._metric_yaw_acceleration_max_rad_s2[ids], yaw_acceleration
        )
        self._metric_previous_yaw_rate_rad_s[ids] = yaw_rate

    def _reward_episode_totals(
        self, ids: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        weights = {
            name: self.reward_manager.get_term_cfg(name).weight
            for name in self.reward_manager.active_terms
        }
        return split_reward_totals(self.reward_manager._episode_sums, weights, ids)

    def _record_completed_episodes(self, ids: torch.Tensor) -> None:
        valid = ids[self.episode_length_buf[ids] > 0]
        if valid.numel() == 0:
            return
        self._update_episode_metrics(valid)
        collision = self.termination_manager.get_term("collision")
        workspace_exit = self.termination_manager.get_term("out_of_bounds")
        reached = self.termination_manager.get_term("goal_reached")
        timeout = self.termination_manager.get_term("time_out")
        action = self.action_manager.get_term("px4_waypoint")
        reward_raw, reward_weighted = self._reward_episode_totals(valid)
        safe_success = successful_episode(
            reached,
            collision,
            workspace_exit,
            self._reward_episode_safe,
        )
        valid_row = {env_id: row for row, env_id in enumerate(valid.tolist())}
        terminal_position = self.scene["robot"].data.root_pos_w - self.scene.env_origins
        terminal_goal_distance = torch.linalg.vector_norm(
            self.goal_w - terminal_position, dim=-1
        )
        for env_id in valid.tolist():
            steps = int(self.episode_length_buf[env_id].item())
            self.completed_records.append(
                {
                    "environment": env_id,
                    "seed": int(self.layout_seed[env_id].item()),
                    "difficulty": int(self.layout_difficulty[env_id].item()),
                    "route_family": self.layout_route_family[env_id],
                    "detour_direction": self.layout_detour_direction[env_id],
                    "narrow_passage": bool(self.layout_narrow_passage[env_id].item()),
                    "direct_path_blocked": bool(
                        self.layout_direct_path_blocked[env_id].item()
                    ),
                    "reference_minimum_clearance_m": (
                        float(self.layout_minimum_route_clearance_m[env_id].item())
                        if math.isfinite(
                            float(self.layout_minimum_route_clearance_m[env_id].item())
                        )
                        else None
                    ),
                    "reference_clearance_known": bool(
                        torch.isfinite(
                            self.layout_minimum_route_clearance_m[env_id]
                        ).item()
                    ),
                    "route_evidence": "grid_nodes_and_0.05_m_segment_samples",
                    "success": bool(safe_success[env_id].item()),
                    "goal_reached": bool(reached[env_id].item()),
                    "episode_safe": bool(self._reward_episode_safe[env_id].item()),
                    "collision": bool(collision[env_id].item()),
                    "workspace_exit": bool(workspace_exit[env_id].item()),
                    "timeout": bool(timeout[env_id].item()),
                    "terminal_position_m": [
                        float(value) for value in terminal_position[env_id].tolist()
                    ],
                    "goal_m": [float(value) for value in self.goal_w[env_id].tolist()],
                    "terminal_goal_distance_m": float(
                        terminal_goal_distance[env_id].item()
                    ),
                    "minimum_clearance_m": float(self._metric_minimum_clearance_m[env_id].item()),
                    "altitude_violation_s": float(self._metric_altitude_violation_s[env_id].item()),
                    "stall_time_s": float(self._metric_stall_time_s[env_id].item()),
                    "path_efficiency": float(
                        self._metric_path_length_m[env_id].item()
                        / max(self._metric_reference_length_m[env_id].item(), 1.0e-6)
                    ),
                    "body_rate_max_rad_s": float(self._metric_body_rate_max_rad_s[env_id].item()),
                    "yaw_acceleration_max_rad_s2": float(
                        self._metric_yaw_acceleration_max_rad_s2[env_id].item()
                    ),
                    "action_saturation_fraction": float(
                        action.action_saturated_total[env_id].item() / max(steps, 1)
                    ),
                    "rewards_raw": {
                        name: float(values[valid_row[env_id]].item())
                        for name, values in reward_raw.items()
                    },
                    "rewards_weighted": {
                        name: float(values[valid_row[env_id]].item())
                        for name, values in reward_weighted.items()
                    },
                }
            )

    def _reset_idx(self, env_ids) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if hasattr(self, "reward_manager"):
            valid = ids[self.episode_length_buf[ids] > 0]
            if valid.numel() > 0:
                reward_raw, reward_weighted = self._reward_episode_totals(valid)
                log = self.extras.setdefault("log", {})
                for name in self.reward_manager.active_terms:
                    log[f"Reward_raw/{name}"] = reward_raw[name].mean()
                    log[f"Reward_weighted/{name}"] = reward_weighted[name].mean()
        if self.cfg.record_episode_records and hasattr(self, "termination_manager"):
            self._record_completed_episodes(ids)
        super()._reset_idx(env_ids)
        layouts = self._select_layouts(ids)

        obstacle_state = self.scene["obstacles"].data.default_object_state[ids].clone()
        obstacle_state[..., :3] = self.scene.env_origins[ids, None, :]
        obstacle_state[..., 2] -= 20.0
        obstacle_state[..., 3:7] = 0.0
        obstacle_state[..., 3] = 1.0
        obstacle_state[..., 7:] = 0.0
        self.obstacle_active[ids] = False
        self.obstacle_centres_w[ids] = 0.0
        self.obstacle_sizes_m[ids] = 0.0

        robot_state = self.scene["robot"].data.default_root_state[ids].clone()
        robot_state[:, 7:] = 0.0
        for row, (env_id, layout) in enumerate(zip(ids.tolist(), layouts, strict=True)):
            self.goal_w[env_id] = torch.tensor(layout.goal_m, device=self.device)
            self.bounds_min_m[env_id] = torch.tensor(layout.bounds_min_m, device=self.device)
            self.bounds_max_m[env_id] = torch.tensor(layout.bounds_max_m, device=self.device)
            robot_state[row, :3] = self.scene.env_origins[env_id] + torch.tensor(
                layout.start_m, device=self.device
            )
            yaw = -math.pi + 2.0 * math.pi * (
                ((self.layout_bank_seed_start + self._layout_reset_count + env_id) % 997) / 997.0
            )
            robot_state[row, 3:7] = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                torch.tensor([yaw], device=self.device),
            )[0]
            for obstacle in layout.obstacles:
                slot = obstacle.slot_index
                if slot < 0:
                    raise RuntimeError("generated obstacle has no physical collider slot")
                centre = torch.tensor(obstacle.center_m, device=self.device)
                size = torch.tensor(obstacle.size_m, device=self.device)
                obstacle_state[row, slot, :3] = self.scene.env_origins[env_id] + centre
                self.obstacle_centres_w[env_id, slot] = centre
                self.obstacle_sizes_m[env_id, slot] = size
                self.obstacle_active[env_id, slot] = True

        self.scene["robot"].write_root_state_to_sim(robot_state, env_ids=ids)
        self.scene["obstacles"].write_object_state_to_sim(obstacle_state, env_ids=ids)
        self._metric_previous_position_w[ids] = robot_state[:, :3]
        self._metric_path_length_m[ids] = 0.0
        self._metric_minimum_clearance_m[ids] = torch.inf
        self._metric_altitude_violation_s[ids] = 0.0
        self._metric_stall_time_s[ids] = 0.0
        self._metric_body_rate_max_rad_s[ids] = 0.0
        self._metric_previous_yaw_rate_rad_s[ids] = 0.0
        self._metric_yaw_acceleration_max_rad_s2[ids] = 0.0
        for env_id, layout in zip(ids.tolist(), layouts, strict=True):
            self._metric_reference_length_m[env_id] = sum(
                math.dist(first, second)
                for first, second in zip(layout.reference_path, layout.reference_path[1:])
            )
        self._disparity.reset(ids)
        self._reward_distance_valid[ids] = False
        self._reward_goal_latched[ids] = False
        self._reward_episode_safe[ids] = True
        self._reward_yaw_valid[ids] = False
        self._reward_yaw_rate_valid[ids] = False
        self._reward_acceleration_valid[ids] = False

    def step(self, action):
        observations, reward, terminated, truncated, extras = super().step(action)
        active = (~(terminated | truncated)).nonzero(as_tuple=False).squeeze(-1)
        self._update_episode_metrics(active)
        position = self.scene["robot"].data.root_pos_w - self.scene.env_origins
        distance = torch.linalg.vector_norm(self.goal_w - position, dim=-1)
        extras.setdefault("log", {}).update(
            {
                "Navigation/goal_distance_m": distance.mean(),
                "Navigation/goal_success": (distance <= self.cfg.goal_tolerance_m).float().mean(),
                "Navigation/difficulty": self.layout_difficulty.float().mean(),
                "Observation/clipped_continuous_fraction": (
                    clipped_continuous_fraction(observations["policy"])
                ),
            }
        )
        return observations, reward, terminated, truncated, extras
