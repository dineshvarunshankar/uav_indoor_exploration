"""Terminal conditions."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from ...inspection import MissionPhase


def collision(
    env: ManagerBasedEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history
    return torch.linalg.vector_norm(forces, dim=-1).amax(dim=(1, 2)) > threshold


def recovery_failed(env: ManagerBasedEnv) -> torch.Tensor:
    return env.mission.recovery_failed


def mission_done(env: ManagerBasedEnv) -> torch.Tensor:
    return env.mission.phase == int(MissionPhase.DONE)


def entry_skill_success(env: ManagerBasedEnv) -> torch.Tensor:
    """The isolated entry episode ends when the aircraft is fully through its opening."""

    if not env.cfg.entry_training.enabled:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    crossed_selected, _ = env.entry_crossing
    return crossed_selected


def unauthorized_crossing(env: ManagerBasedEnv) -> torch.Tensor:
    """The aircraft is behind the facade without entry permission."""

    if not env.cfg.entry_training.enabled:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.unauthorized_crossing_now
