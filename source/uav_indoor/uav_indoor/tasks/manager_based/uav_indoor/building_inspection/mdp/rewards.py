"""Rewards."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

from ...inspection import MissionPhase, MissionStage
from ..reward_features import (
    action_change,
    altitude_violation,
    body_rate_cost,
    box_center_error,
    collision_features,
    fixed_pose_errors,
    gate_final_floor_completion,
    potential_progress,
    visibility_signal,
    yaw_acceleration_cost,
)


def _robot_position(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_pos_w - env.scene.env_origins


def _delta(env: ManagerBasedEnv, name: str, value: torch.Tensor) -> torch.Tensor:
    """Increase of a monotone counter since the last step, paid as a one-off event."""

    previous = getattr(env, name, None)
    if previous is None:
        previous = value.clone()
    gained = (value - previous).clamp_min(0.0)
    setattr(env, name, value.clone())
    return gained / env.step_dt






def _max_contact_force(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.linalg.vector_norm(sensor.data.net_forces_w_history, dim=-1).amax(dim=(1, 2))


def collision_penalty(
    env: ManagerBasedEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Collision event rate; Isaac Lab integrates it to one event cost."""
    event, _ = collision_features(_max_contact_force(env, sensor_cfg), threshold)
    return event / env.step_dt


def collision_severity_penalty(
    env: ManagerBasedEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Contact-force excess while a collision is active."""
    _, severity = collision_features(_max_contact_force(env, sensor_cfg), threshold)
    return severity


def altitude_penalty(env: ManagerBasedEnv, minimum_m: float = 2.5) -> torch.Tensor:
    """Metres below the fixed launch-ground clearance for this milestone."""
    return altitude_violation(_robot_position(env)[:, 2], minimum_m)


def body_rate_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Squared roll, pitch, and yaw body rates."""
    return body_rate_cost(env.scene["robot"].data.root_ang_vel_b)


def yaw_acceleration_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Squared yaw acceleration with history isolated from mission telemetry."""
    current = env.scene["robot"].data.root_ang_vel_b[:, 2]
    previous = getattr(env, "_reward_previous_yaw_rate", current).clone()
    valid = getattr(
        env, "_reward_yaw_history_valid",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    cost = yaw_acceleration_cost(current, previous, env.step_dt)
    env._reward_previous_yaw_rate = current.clone()
    env._reward_yaw_history_valid = torch.ones_like(valid)
    return torch.where(valid, cost, torch.zeros_like(cost))


def recovery_failed_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """One full event cost when the protected exit route cannot be completed."""

    return env.mission.recovery_failed.float() / env.step_dt








def action_rate_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    term = env.action_manager.get_term("px4_waypoint")
    return action_change(term.raw_actions, env.action_manager.prev_action)




def transit_completion(env: ManagerBasedEnv) -> torch.Tensor:
    """One event reward when the first curriculum stage reaches the standoff."""

    completed = (
        (env.mission.phase == int(MissionPhase.DONE)) & (env.mission.stage == int(MissionStage.TRANSIT_ONLY))
    ).float()
    return _delta(env, "_previous_transit_completion", completed)


def reference_progress(env: ManagerBasedEnv) -> torch.Tensor:
    """Rate of decrease in route distance, reset by phase or target changes."""
    distance = torch.linalg.vector_norm(env.mission_reference.position_w - _robot_position(env), dim=-1)
    phase = env.mission_reference.phase
    previous = getattr(env, "_reward_previous_distance", distance).clone()
    previous_phase = getattr(env, "_reward_previous_reference_phase", phase).clone()
    valid = getattr(
        env, "_reward_reference_history_valid",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    progress, env._reward_previous_distance, env._reward_previous_reference_phase = potential_progress(
        previous, distance, previous_phase, phase, env.step_dt
    )
    target_changed = getattr(
        env, "active_target_changed",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    reset = ~valid | target_changed
    env._reward_reference_history_valid = torch.ones_like(valid)
    return torch.where(reset, torch.zeros_like(progress), progress).clamp(
        -1.0 / env.step_dt, 1.0 / env.step_dt
    )


def visual_alignment_progress(env: ManagerBasedEnv) -> torch.Tensor:
    """Rate of decrease in selected live-box centre error."""
    phase = env.mission_reference.phase
    active_phase = (phase == int(MissionPhase.APPROACH)) | (phase == int(MissionPhase.VALIDATE)) | (
        phase == int(MissionPhase.ENTRY)
    )
    valid = env.active_opening_box_valid & active_phase
    error = box_center_error(env.active_opening_box)
    previous = getattr(env, "_reward_previous_box_error", error).clone()
    previous_phase = getattr(env, "_reward_previous_box_phase", phase).clone()
    previous_valid = getattr(
        env, "_reward_box_history_valid",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    progress, env._reward_previous_box_error, env._reward_previous_box_phase = potential_progress(
        previous, error, previous_phase, phase, env.step_dt
    )
    reset = ~valid | ~previous_valid | env.active_target_changed
    env._reward_box_history_valid = valid.clone()
    return torch.where(reset, torch.zeros_like(progress), progress)


def reference_yaw_progress(env: ManagerBasedEnv) -> torch.Tensor:
    """Rate of decrease in wrapped yaw error, without a reward for merely holding."""
    phase = env.mission_reference.phase
    error = wrap_to_pi(env.mission_reference.yaw_w - env.robot_yaw_w).abs()
    previous = getattr(env, "_reward_previous_yaw_error", error).clone()
    previous_phase = getattr(env, "_reward_previous_yaw_phase", phase).clone()
    valid = getattr(
        env, "_reward_yaw_progress_valid",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    progress, env._reward_previous_yaw_error, env._reward_previous_yaw_phase = potential_progress(
        previous, error, previous_phase, phase, env.step_dt
    )
    reset = ~valid | env.active_target_changed
    env._reward_yaw_progress_valid = torch.ones_like(valid)
    return torch.where(reset, torch.zeros_like(progress), progress)


def opening_visibility(env: ManagerBasedEnv) -> torch.Tensor:
    """Raw selected-track visibility during opening servoing phases."""
    phase = env.mission_reference.phase
    active = (phase == int(MissionPhase.APPROACH)) | (phase == int(MissionPhase.VALIDATE)) | (
        phase == int(MissionPhase.ENTRY)
    )
    return visibility_signal(env.active_opening_box_valid) * active.float()


def fixed_position_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Position error during VALIDATE and OBSERVE; metres remain unscaled."""
    position, _ = fixed_pose_errors(
        _robot_position(env), env.mission_reference.position_w, env.robot_yaw_w, env.mission_reference.yaw_w
    )
    phase = env.mission_reference.phase
    active = (phase == int(MissionPhase.VALIDATE)) | (phase == int(MissionPhase.OBSERVE))
    return position * active.float()


def fixed_yaw_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Yaw error during VALIDATE and OBSERVE; radians remain unscaled."""
    _, yaw = fixed_pose_errors(
        _robot_position(env), env.mission_reference.position_w, env.robot_yaw_w, env.mission_reference.yaw_w
    )
    phase = env.mission_reference.phase
    active = (phase == int(MissionPhase.VALIDATE)) | (phase == int(MissionPhase.OBSERVE))
    return yaw * active.float()



def survey_coverage(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward each scan station completed, whichever way along the wall it lies."""

    return _delta(env, "_previous_station", env.mission.stations_scanned.float())




def authorized_crossing(env: ManagerBasedEnv) -> torch.Tensor:
    """One reward for each authorized entry that reaches the inside pose."""

    observing = (env.mission_reference.phase == int(MissionPhase.OBSERVE)).float()
    return _delta(env, "_previous_observing", observing)




def exit_progress(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward each breadcrumb consumed on the way out."""

    buffer = env.mission.breadcrumbs
    sealed = buffer.exit_index >= 0
    walked = torch.where(sealed, (buffer.count - 1 - buffer.exit_index).clamp_min(0), torch.zeros_like(buffer.count))
    return _delta(env, "_previous_consumed", walked.float())


def opening_completion(env: ManagerBasedEnv) -> torch.Tensor:
    inspected = env.true_inspected_openings.sum(dim=1).float()
    return _delta(env, "_previous_inspected", inspected)


def safe_exit_time_bonus(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward a completed safe exit."""

    inspected = env.true_inspected_openings.sum(dim=1).float()
    safe_exit = _delta(env, "_previous_safe_exit", inspected)
    full_stage = env.mission.stage == int(MissionStage.FULL)
    time_remaining = (1.0 - env.episode_length_buf.float() / env.max_episode_length).clamp(0.0, 1.0)
    return safe_exit * full_stage.float() * time_remaining


def floor_completion(env: ManagerBasedEnv) -> torch.Tensor:
    floor_delta = _delta(env, "_previous_floor", env.mission.floor.float())
    mission_done = env.mission.phase == int(MissionPhase.DONE)
    return gate_final_floor_completion(floor_delta, mission_done, env.mission.mission_success)


def entry_crossing_bonus(env: ManagerBasedEnv) -> torch.Tensor:
    """Reward an authorized crossing once."""

    if not env.cfg.entry_training.enabled:
        return torch.zeros(env.num_envs, device=env.device)
    crossed, _ = env.entry_crossing
    authorized = env.mission_reference.entry_authorized
    event = _delta(env, "_previous_entry_crossing", (crossed & authorized).float())
    time_remaining = (1.0 - env.episode_length_buf.float() / env.max_episode_length).clamp(0.0, 1.0)
    return event * time_remaining


def unauthorized_crossing_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Penalize unauthorized facade crossings."""

    return env.unauthorized_crossing_now.float() / env.step_dt


def wrong_opening_penalty(env: ManagerBasedEnv) -> torch.Tensor:
    """Going through a different way in is not the skill being trained."""

    if not env.cfg.entry_training.enabled:
        return torch.zeros(env.num_envs, device=env.device)
    _, elsewhere = env.entry_crossing
    return _delta(env, "_previous_wrong_opening", elsewhere.float())
