from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from ..inspection import BuildingMissionSupervisor, MissionPhase

if TYPE_CHECKING:
    pass


DEFAULT_STANDOFF_ENV = torch.tensor((48.52, -38.906, 0.855), dtype=torch.float32)


def _asset(env, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name]


def _robot_position_env(env, asset) -> torch.Tensor:
    return asset.data.root_pos_w - env.scene.env_origins


def _robot_yaw(asset) -> torch.Tensor:
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return math_utils.wrap_to_pi(yaw)


def _as_env_tensor(env, name: str, shape: tuple[int, ...], fill: float = 0.0, dtype=torch.float32) -> torch.Tensor:
    value = getattr(env, name, None)
    if value is None:
        return torch.full((env.num_envs, *shape), fill, device=env.device, dtype=dtype)
    return value.to(device=env.device, dtype=dtype)


def _as_env_bool(env, name: str, shape: tuple[int, ...], fill: bool = False) -> torch.Tensor:
    value = getattr(env, name, None)
    if value is None:
        return torch.full((env.num_envs, *shape), fill, device=env.device, dtype=torch.bool)
    return value.to(device=env.device, dtype=torch.bool)


def get_building_targets(env) -> torch.Tensor:
    """Return the active supplied standoff target in environment coordinates."""

    if not hasattr(env, "building_target_env"):
        target = DEFAULT_STANDOFF_ENV.to(device=env.device).repeat(env.num_envs, 1)
        env.building_target_env = target
    return env.building_target_env


def _mission_supervisor(env) -> BuildingMissionSupervisor:
    if not hasattr(env, "building_mission"):
        env.building_mission = BuildingMissionSupervisor(env.num_envs, env.device)
        env._building_mission_step = -1
    return env.building_mission


def reset_building_mission(env, env_ids: torch.Tensor, building_targets: torch.Tensor | None = None) -> None:
    """Reset the deterministic inspection mission for selected environments."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    targets = get_building_targets(env)
    if building_targets is None:
        targets[env_ids] = DEFAULT_STANDOFF_ENV.to(device=env.device)
    else:
        targets[env_ids] = building_targets.to(device=env.device, dtype=torch.float32)
    _mission_supervisor(env).reset(env_ids)
    env._building_mission_step = -1
    for name in (
        "_prev_mission_ref_distance",
        "_prev_stations_scanned",
        "_prev_terminal_count",
        "_prev_observation_progress",
    ):
        if hasattr(env, name):
            delattr(env, name)


def update_building_mission(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Advance the mission supervisor at most once per manager step."""

    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_building_mission_step", -1) == step and hasattr(env, "building_mission_reference"):
        return env.building_mission_reference

    asset = _asset(env, asset_cfg)
    supervisor = _mission_supervisor(env)
    robot_position_w = asset.data.root_pos_w
    robot_yaw_w = _robot_yaw(asset)
    building_standoff_w = get_building_targets(env) + env.scene.env_origins

    detections = int(getattr(env, "building_detection_slots", 1))
    opening_origin_w = getattr(env, "opening_origin_w", robot_position_w)
    opening_ray_w = _as_env_tensor(env, "opening_ray_w", (detections, 3), fill=0.0)
    if not hasattr(env, "opening_ray_w"):
        opening_ray_w[..., 0] = 1.0
    opening_angular_wh = _as_env_tensor(env, "opening_angular_wh", (detections, 2), fill=0.0)
    opening_track_id = _as_env_tensor(env, "opening_track_id", (detections,), fill=-1, dtype=torch.long)
    opening_box_xywhn = _as_env_tensor(env, "opening_box_xywhn", (detections, 4), fill=0.0)
    opening_confidence = _as_env_tensor(env, "opening_confidence", (detections,), fill=0.0)
    opening_valid = _as_env_bool(env, "opening_valid", (detections,), fill=False)

    person_slots = int(getattr(env, "person_detection_slots", 1))
    person_track_id = _as_env_tensor(env, "person_track_id", (person_slots,), fill=-1.0, dtype=torch.long)
    person_box_xywhn = _as_env_tensor(env, "person_box_xywhn", (person_slots, 4), fill=0.0)
    person_confidence = _as_env_tensor(env, "person_confidence", (person_slots,), fill=0.0)
    person_age_s = _as_env_tensor(env, "person_age_s", (person_slots,), fill=0.0)
    person_valid = _as_env_bool(env, "person_valid", (person_slots,), fill=False)

    tof_valid = _as_env_bool(env, "tof_valid", (), fill=False)
    tof_clear = _as_env_bool(env, "tof_clear", (), fill=False)
    clearance_m = _as_env_tensor(env, "clearance_m", (), fill=0.0)
    clearance_known = _as_env_bool(env, "clearance_known", (), fill=False)
    avoidance_active = _as_env_bool(env, "avoidance_active", (), fill=False)

    dt = float(getattr(env, "step_dt", getattr(env, "physics_dt", 1.0 / 60.0)))
    time_s = float(step * dt)
    reference = supervisor.update(
        dt=dt,
        robot_position_w=robot_position_w,
        robot_yaw_w=robot_yaw_w,
        building_standoff_w=building_standoff_w,
        opening_origin_w=opening_origin_w,
        opening_ray_w=opening_ray_w,
        opening_angular_wh=opening_angular_wh,
        opening_track_id=opening_track_id,
        opening_box_xywhn=opening_box_xywhn,
        opening_confidence=opening_confidence,
        opening_valid=opening_valid,
        opening_box_centered=_as_env_bool(env, "active_opening_box_centered", (), fill=False),
        person_track_id=person_track_id,
        person_box_xywhn=person_box_xywhn,
        person_confidence=person_confidence,
        person_age_s=person_age_s,
        person_valid=person_valid,
        tof_valid=tof_valid,
        tof_clear=tof_clear,
        clearance_m=clearance_m,
        clearance_known=clearance_known,
        avoidance_active=avoidance_active,
        time_s=time_s,
    )
    env.building_mission_reference = reference
    env._building_mission_step = step
    return reference


def mission_reference_observation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Actor-visible mission reference, ledger summary, and route state."""

    asset = _asset(env, asset_cfg)
    reference = update_building_mission(env, asset_cfg)
    offset_w = reference.position_w - asset.data.root_pos_w
    offset_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, offset_w)
    distance = torch.linalg.vector_norm(offset_b, dim=-1, keepdim=True)
    direction = offset_b / distance.clamp_min(1.0e-6)
    yaw = _robot_yaw(asset)
    yaw_error = math_utils.wrap_to_pi(reference.yaw_w - yaw)[:, None]
    mission = _mission_supervisor(env).observation(asset.data.root_pos_w)
    flags = torch.stack(
        (
            reference.entry_authorized.float(),
            reference.crossing_opening.float(),
            reference.target_valid.float(),
            reference.detector_enabled.float(),
            reference.accepts_proposals.float(),
            reference.observation_progress,
        ),
        dim=-1,
    )
    return torch.cat((direction, distance, yaw_error, flags, mission), dim=-1)


def disparity_history_observation(
    env, history: int = 4, height: int = 12, width: int = 16
) -> torch.Tensor:
    """Return flattened relative disparity history."""

    value = getattr(env, "disparity_history", None)
    if value is None:
        return torch.zeros(env.num_envs, history * height * width, device=env.device)
    return value.to(device=env.device, dtype=torch.float32).reshape(env.num_envs, -1)


def tof_corridor_observation(env) -> torch.Tensor:
    """ToF corridor evidence as valid flag and clipped range."""

    valid = _as_env_bool(env, "tof_valid", (), fill=False).float()[:, None]
    range_m = _as_env_tensor(env, "tof_range_m", (), fill=0.0)[:, None].clamp(0.0, 10.0) / 10.0
    return torch.cat((valid, range_m), dim=-1)


def mission_privileged_observation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Critic-only simulator/mission truth summary."""

    reference = update_building_mission(env, asset_cfg)
    supervisor = _mission_supervisor(env)
    counts = torch.stack(
        (
            supervisor.floor.float(),
            supervisor.facade_index.float(),
            supervisor.stations_scanned.float(),
            supervisor.recovery_failed.float(),
            (reference.phase == int(MissionPhase.DONE)).float(),
        ),
        dim=-1,
    )
    return torch.cat((counts, reference.phase.float()[:, None]), dim=-1)


def _reference_distance(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = _asset(env, asset_cfg)
    reference = update_building_mission(env, asset_cfg)
    return torch.linalg.vector_norm(reference.position_w - asset.data.root_pos_w, dim=-1)


def mission_progress_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    distance = _reference_distance(env, asset_cfg)
    if not hasattr(env, "_prev_mission_ref_distance"):
        env._prev_mission_ref_distance = distance.clone()
        return torch.zeros_like(distance)
    progress = env._prev_mission_ref_distance - distance
    env._prev_mission_ref_distance = distance.clone()
    return progress


def survey_coverage_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    update_building_mission(env, asset_cfg)
    scanned = _mission_supervisor(env).stations_scanned.float()
    if not hasattr(env, "_prev_stations_scanned"):
        env._prev_stations_scanned = scanned.clone()
        return torch.zeros_like(scanned)
    reward = (scanned - env._prev_stations_scanned).clamp_min(0.0)
    env._prev_stations_scanned = scanned.clone()
    return reward


def opening_validation_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    update_building_mission(env, asset_cfg)
    ledger = _mission_supervisor(env).ledger
    terminal = ledger.terminal.sum(dim=1).float()
    if not hasattr(env, "_prev_terminal_count"):
        env._prev_terminal_count = terminal.clone()
        return torch.zeros_like(terminal)
    reward = (terminal - env._prev_terminal_count).clamp_min(0.0)
    env._prev_terminal_count = terminal.clone()
    return reward


def entry_exit_recovery_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    update_building_mission(env, asset_cfg)
    supervisor = _mission_supervisor(env)
    return supervisor.breadcrumbs.exit_complete.float() - supervisor.recovery_failed.float()


def person_observation_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    reference = update_building_mission(env, asset_cfg)
    progress = reference.observation_progress
    if not hasattr(env, "_prev_observation_progress"):
        env._prev_observation_progress = progress.clone()
        return torch.zeros_like(progress)
    reward = (progress - env._prev_observation_progress).clamp_min(0.0)
    env._prev_observation_progress = progress.clone()
    return reward


def building_complete_reward(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    reference = update_building_mission(env, asset_cfg)
    return (reference.phase == int(MissionPhase.DONE)).float()


def clearance_safety_reward(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), margin_m: float = 1.0
) -> torch.Tensor:
    clearance = _as_env_tensor(env, "clearance_m", (), fill=margin_m)
    return -((margin_m - clearance).clamp_min(0.0) ** 2)


def building_mission_done(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    reference = update_building_mission(env, asset_cfg)
    return reference.phase == int(MissionPhase.DONE)
