"""Pure cumulative curriculum policy for random-goal navigation."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

MAX_DIFFICULTY = 3


@dataclass(frozen=True)
class CurriculumState:
    current_difficulty: int = 0
    rehearsal_boosts: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class EvaluationResult:
    difficulty: int
    episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    altitude_violation_rate: float
    stalled_rate: float


def sample_difficulties(state: CurriculumState, count: int, seed: int) -> list[int]:
    """Sample the current stage most often while retaining every passed stage."""

    if count < 0:
        raise ValueError("count must be non-negative")
    current = max(0, min(MAX_DIFFICULTY, state.current_difficulty))
    if current == 0:
        return [0] * count
    stages = list(range(current + 1))
    weights = [0.30 / current] * current + [0.70]
    weights = [weight + state.rehearsal_boosts[stage] for stage, weight in zip(stages, weights)]
    rng = random.Random(seed)
    sample = stages.copy() if count >= len(stages) else []
    sample.extend(rng.choices(stages, weights=weights, k=count - len(sample)))
    rng.shuffle(sample)
    return sample


def _passes(result: EvaluationResult) -> bool:
    return (
        result.episodes >= 32
        and result.success_rate == 1.0
        and result.collision_rate == 0.0
        and result.timeout_rate == 0.0
        and result.altitude_violation_rate == 0.0
        and result.stalled_rate == 0.0
    )


def update_curriculum(
    state: CurriculumState, results: list[EvaluationResult]
) -> CurriculumState:
    """Promote once only when current and earlier held-out gates all pass."""

    by_stage = {result.difficulty: result for result in results}
    required = range(state.current_difficulty + 1)
    failed = [stage for stage in required if stage not in by_stage or not _passes(by_stage[stage])]
    if failed:
        boosts = list(state.rehearsal_boosts)
        for stage in failed:
            boosts[stage] = min(boosts[stage] + 0.10, 1.0)
        return replace(state, rehearsal_boosts=tuple(boosts))
    if state.current_difficulty >= MAX_DIFFICULTY:
        return state
    return CurriculumState(
        current_difficulty=state.current_difficulty + 1,
        rehearsal_boosts=state.rehearsal_boosts,
    )
