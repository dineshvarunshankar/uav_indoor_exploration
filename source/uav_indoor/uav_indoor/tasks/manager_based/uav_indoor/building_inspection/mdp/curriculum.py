"""Curriculum over transit, survey, validation, and full entry."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from ...inspection import MissionStage


def _stage_success(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Return the active stage success condition."""

    stage = env.mission.stage[env_ids]
    transited = env._transit_arrived[env_ids]
    surveyed = env.true_recorded_openings[env_ids].any(dim=1)
    validated = env.true_validated_openings[env_ids].any(dim=1)
    inspected = env.true_inspected_openings[env_ids].any(dim=1)
    return torch.where(
        stage == int(MissionStage.TRANSIT_ONLY),
        transited,
        torch.where(
            stage == int(MissionStage.SURVEY_ONLY),
            surveyed,
            torch.where(stage == int(MissionStage.VALIDATE_ONLY), validated, inspected),
        ),
    )


def _stage_failure(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Count safety failures, not successful or ordinary episode termination."""

    return (
        env.termination_manager.get_term("collision")[env_ids]
        | env.termination_manager.get_term("recovery_failed")[env_ids]
    )


def mission_stage(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    promote_success_rate: float = 0.6,
    demote_failure_rate: float = 0.3,
    minimum_episodes: int = 32,
    window: int = 128,
) -> float:
    """Promote after repeated stage success."""

    if env.cfg.mission.lock_stage:
        return float(env.mission.stage[0].item())

    episode_stage = env.mission.stage[env_ids].clone()
    success = _stage_success(env, env_ids).float()
    failure = _stage_failure(env, env_ids).float()
    finished = env.episode_length_buf[env_ids] > 0

    if not hasattr(env, "_curriculum_stage"):
        env._curriculum_stage = int(env.mission.stage[0].item())
        env._stage_successes = torch.zeros(window, device=env.device)
        env._stage_failures = torch.zeros(window, device=env.device)
        env._stage_index = 0
        env._stage_episode_count = 0

    eligible = finished & (episode_stage == env._curriculum_stage)
    for value, hit in zip(success[eligible], failure[eligible], strict=True):
        slot = env._stage_index % window
        env._stage_successes[slot] = value
        env._stage_failures[slot] = hit
        env._stage_index += 1
        env._stage_episode_count = min(env._stage_episode_count + 1, window)

    stage_changed = False
    if env._stage_episode_count >= minimum_episodes:
        sample = slice(0, env._stage_episode_count)
        success_rate = env._stage_successes[sample].mean()
        failure_rate = env._stage_failures[sample].mean()
        if failure_rate >= demote_failure_rate and env._curriculum_stage > int(MissionStage.TRANSIT_ONLY):
            env._curriculum_stage -= 1
            stage_changed = True
        elif (
            success_rate >= promote_success_rate
            and failure_rate < demote_failure_rate
            and env._curriculum_stage < int(MissionStage.FULL)
        ):
            env._curriculum_stage += 1
            stage_changed = True

    if stage_changed:
        env._stage_successes.zero_()
        env._stage_failures.zero_()
        env._stage_index = 0
        env._stage_episode_count = 0

    reset_stage = torch.full((env_ids.numel(),), env._curriculum_stage, dtype=torch.long, device=env.device)
    env.mission.set_stage(reset_stage, env_ids)
    return float(env._curriculum_stage)
