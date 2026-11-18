"""Episode setup: spawn pose and building target."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv

from ..entry_training import sample_entry_start
from ..truth import OPENINGS


def reset_to_spawn_zone(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Place the aircraft at a random point in the approach box, outside the building."""

    if env.cfg.entry_training.enabled:
        reset_to_entry_start(env, env_ids)
        return

    box = torch.tensor(env.cfg.approach_box_w, device=env.device)
    fraction = torch.rand(len(env_ids), 3, device=env.device)
    position = box[0::2] + fraction * (box[1::2] - box[0::2])

    asset = env.scene["robot"]
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] = position + env.scene.env_origins[env_ids]
    root_state[:, 7:] = 0.0
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def assign_building_target(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Set the local building standoff and load the truth openings used for scoring."""

    openings = OPENINGS.to(env.device)
    if env.cfg.entry_training.enabled:
        normal = torch.tensor(env.cfg.entry_training.truth_facade_normal_w, device=env.device)
        env.building_standoff_w[env_ids] = (
            env.entry_target_w[env_ids] + normal * env.cfg.mission.standoff_m
        )
    else:
        env.building_standoff_w[env_ids] = torch.tensor(env.cfg.building_standoff_w, device=env.device)
    count = min(openings.shape[0], env.opening_truth_w.shape[1])
    env.opening_truth_w[env_ids] = 0.0
    env.opening_truth_w[env_ids, :count] = openings[:count]
    env.truth_count[env_ids] = count


def reset_to_entry_start(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Start the isolated entry episode in front of one opening chosen by truth."""

    cfg = env.cfg.entry_training
    openings = OPENINGS.to(env.device)
    count = env.num_envs if env_ids is None else len(env_ids)
    chosen = torch.randint(openings.shape[0], (count,), device=env.device)
    target = openings[chosen]
    normal = torch.tensor(env.cfg.entry_training.truth_facade_normal_w, device=env.device).expand(count, 3)

    position, yaw = sample_entry_start(
        target, normal, cfg, float(env.entry_start_width), torch.rand(count, 4, device=env.device)
    )
    env.entry_target_w[env_ids] = target
    env.entry_target_index[env_ids] = chosen
    env.entry_start_w[env_ids] = position

    asset = env.scene["robot"]
    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] = position + env.scene.env_origins[env_ids]
    root_state[:, 3] = torch.cos(0.5 * yaw)
    root_state[:, 4:6] = 0.0
    root_state[:, 6] = torch.sin(0.5 * yaw)
    root_state[:, 7:] = 0.0
    asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
