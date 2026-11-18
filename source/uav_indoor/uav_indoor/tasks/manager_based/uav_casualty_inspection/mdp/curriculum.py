"""Staged inspection mission curriculum."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from ..mission import FINAL_STAGE


def staged_mission_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    success_rate: float = 0.6,
    minimum_transitions: int = 20_000,
) -> torch.Tensor:
    """Promote based on recent mission completion."""

    del env_ids
    if env.cfg.auto_curriculum and env.cfg.curriculum_stage < FINAL_STAGE:
        transitions_since_promotion = (
            env.common_step_counter - env.stage_started_step
        ) * env.num_envs
        if (
            transitions_since_promotion >= minimum_transitions
            and env.episode_success_rate >= success_rate
        ):
            env.cfg.curriculum_stage += 1
            env.on_curriculum_promoted()
    return torch.tensor(float(env.cfg.curriculum_stage), device=env.device)
