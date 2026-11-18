"""Frozen stage-one weights from the recorded three-control measurement."""

from __future__ import annotations

from collections.abc import Mapping

STAGE_ONE_REWARD_WEIGHTS: dict[str, float] = {
    "goal_progress": 5.0,
    "goal_reached": 100.0,
    "collision_event": -150.0,
    "collision_severity": -2.0,
    "workspace_event": -150.0,
    "approach_danger": -4.0,
    "altitude_violation": -20.0,
    "action_change": -0.05,
    "body_rate": -0.01,
    "yaw_acceleration": -0.002,
    "yaw_progress": 1.0,
}


DENSE_SAFETY_REWARD_WEIGHTS: dict[str, float] = {
    "clearance_margin": -2.0,
    "altitude_margin": -4.0,
}


DEPTHNAV_REWARD_WEIGHTS: dict[str, float] = {
    "collision_avoidance": -6.0,
    "yaw_alignment": 1.0,
    "physical_acceleration": -0.01,
    "physical_jerk": -0.001,
}


def weighted_return(raw_integrated_terms: Mapping[str, float]) -> float:
    """Return the weighted episode value for one complete raw-term record."""

    missing = set(STAGE_ONE_REWARD_WEIGHTS) - set(raw_integrated_terms)
    if missing:
        raise ValueError(f"raw reward record is missing {sorted(missing)}")
    return sum(
        STAGE_ONE_REWARD_WEIGHTS[name] * float(raw_integrated_terms[name])
        for name in STAGE_ONE_REWARD_WEIGHTS
    )
