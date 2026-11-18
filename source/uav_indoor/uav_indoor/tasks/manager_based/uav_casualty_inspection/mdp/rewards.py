"""Phase-aware inspection mission rewards and terminal conditions."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from ..mission import MissionPhase


def local_reference_progress(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    position = env.scene[asset_cfg.name].data.root_pos_w
    distance = torch.linalg.vector_norm(env.mission_reference.position_w - position, dim=-1)
    if not hasattr(env, "_previous_reference_distance"):
        env._previous_reference_distance = distance.clone()
    if not hasattr(env, "_previous_reference_phase"):
        env._previous_reference_phase = env.mission_reference.phase.clone()
    progress = env._previous_reference_distance - distance
    phase_changed = env._previous_reference_phase != env.mission_reference.phase
    progress = torch.where(phase_changed, torch.zeros_like(progress), progress)
    env._previous_reference_distance = distance.clone()
    env._previous_reference_phase = env.mission_reference.phase.clone()
    return progress.clamp(-1.0, 1.0)


def reference_tracking_exp(
    env: ManagerBasedEnv,
    std: float = 12.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward proximity to the reference."""

    position = env.scene[asset_cfg.name].data.root_pos_w
    distance = torch.linalg.vector_norm(env.mission_reference.position_w - position, dim=-1)
    return torch.exp(-distance.square() / (std * std))


def reference_heading_exp(
    env: ManagerBasedEnv,
    std: float = 1.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bonus that peaks on the reference heading and decays to zero."""

    _, _, yaw = euler_xyz_from_quat(env.scene[asset_cfg.name].data.root_quat_w)
    error = wrap_to_pi(env.mission_reference.yaw_w - yaw)
    return torch.exp(-error.square() / (std * std))


def clearance_ttc(env: ManagerBasedEnv, safe_clearance_m: float = 1.0) -> torch.Tensor:
    clearance = env.minimum_clearance
    clearance_risk = torch.clamp((safe_clearance_m - clearance) / safe_clearance_m, 0.0, 1.0)
    action = env.action_manager.get_term("px4_velocity").safe_actions
    closing = torch.relu(action[:, 0])
    ttc = clearance / closing.clamp_min(0.1)
    ttc_risk = torch.clamp((2.0 - ttc) / 2.0, 0.0, 1.0)
    return clearance_risk + 0.5 * ttc_risk


def target_visibility(env: ManagerBasedEnv) -> torch.Tensor:
    tracks = env.perception.tracker
    visible = (tracks.valid & (tracks.confidence > 0.25)).any(dim=1)
    phase = env.mission_reference.phase
    relevant = (phase >= int(MissionPhase.OBSERVE)) & (phase < int(MissionPhase.COMPLETE))
    return (visible & relevant).float()


def face_acquisition(env: ManagerBasedEnv) -> torch.Tensor:
    tracks = env.perception.tracker
    face = (tracks.valid & (tracks.face_confidence >= 0.35)).any(dim=1)
    phase = env.mission_reference.phase
    relevant = (phase >= int(MissionPhase.SEARCH_FACE)) & (phase <= int(MissionPhase.RECORD))
    return (face & relevant).float()


def staging_accuracy(
    env: ManagerBasedEnv,
    std: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    phase = env.mission_reference.phase
    relevant = (phase == int(MissionPhase.STAGE)) | (phase == int(MissionPhase.RECORD))
    position = env.scene[asset_cfg.name].data.root_pos_w
    error = torch.linalg.vector_norm(env.mission_reference.position_w - position, dim=-1)
    return torch.exp(-error.square() / (std * std)) * relevant.float()


def recording_quality(env: ManagerBasedEnv) -> torch.Tensor:
    return env.recording_valid.float() * env.mission_reference.recording.float()


def arc_framing_lower_quantile(env: ManagerBasedEnv) -> torch.Tensor:
    return env.arc_frame_lower_quantile * env.mission_reference.recording.float()


def mission_phase_progress(env: ManagerBasedEnv) -> torch.Tensor:
    """Pay each forward step of the mission so credit does not have to cross the whole episode."""

    return env.mission_progress_delta


def mission_completion(env: ManagerBasedEnv) -> torch.Tensor:
    done = env.mission_reference.phase == int(MissionPhase.DONE)
    return (done & ~env.supervisor.failed).float()


def mission_failure(env: ManagerBasedEnv) -> torch.Tensor:
    return env.supervisor.failed.float()


def shield_intervention(env: ManagerBasedEnv) -> torch.Tensor:
    return env.action_manager.get_term("px4_velocity").shield_intervened.float()


def action_rate_l2(env: ManagerBasedEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def collision(
    env: ManagerBasedEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history
    return torch.any(torch.max(torch.linalg.vector_norm(forces, dim=-1), dim=1)[0] > threshold, dim=1)


def mission_done(env: ManagerBasedEnv) -> torch.Tensor:
    return env.mission_reference.phase == int(MissionPhase.DONE)


def mission_failed(env: ManagerBasedEnv) -> torch.Tensor:
    """End the episode when the supervisor gives up."""

    return env.supervisor.failed


def out_of_bounds(
    env: ManagerBasedEnv,
    radius_m: float = 70.0,
    max_height_m: float = 30.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate above the altitude ceiling."""

    position = env.scene[asset_cfg.name].data.root_pos_w - env.scene.env_origins
    return (torch.linalg.vector_norm(position[:, :2], dim=-1) > radius_m) | (
        position[:, 2] > max_height_m
    )
