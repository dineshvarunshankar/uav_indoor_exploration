#!/usr/bin/env python3
"""Gate an entry-skill run on crossing success and on every safety result.

Two profiles share one metric set. ``exploratory`` accepts a partly trained skill so that
learning can be judged; ``release`` is the bar a deployable skill has to clear. Neither
profile tolerates a collision, an unauthorized crossing, or a crossing of the wrong
opening: those are safety results, not learning progress, and they do not improve with
more iterations.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Required evaluation metrics.
REQUIRED_METRICS = (
    "Entry/success_rate",
    "Entry/collision_rate",
    "Entry/unauthorized_crossing_rate",
    "Entry/wrong_opening_crossing_rate",
    "Entry/tof_veto_count",
    "Entry/crossing_time_s",
    "Entry/closest_clearance_m",
    "Entry/target_visible_rate",
    "Entry/target_reacquired_rate",
)

PROFILES = {
    "exploratory": {"min_success_rate": 0.30, "min_target_visible_rate": 0.50},
    "release": {"min_success_rate": 0.95, "min_target_visible_rate": 0.90},
}


def evaluate_metrics(
    metrics: dict[str, float],
    min_success_rate: float,
    min_target_visible_rate: float,
    max_crossing_time_s: float = 30.0,
) -> list[str]:
    """Return one concise failure for each failed condition."""

    failures: list[str] = []

    for tag in REQUIRED_METRICS:
        value = metrics.get(tag)
        if value is None or not math.isfinite(value):
            failures.append(f"{tag} is missing or not finite, got {value}")

    def value_of(tag: str) -> float | None:
        value = metrics.get(tag)
        return value if value is not None and math.isfinite(value) else None

    success = value_of("Entry/success_rate")
    if success is not None and success < min_success_rate:
        failures.append(f"entry success too low: {success:.3f} < {min_success_rate}")

    for tag, label in (
        ("Entry/collision_rate", "collision"),
        ("Entry/unauthorized_crossing_rate", "unauthorized crossing"),
        ("Entry/wrong_opening_crossing_rate", "wrong-opening crossing"),
    ):
        rate = value_of(tag)
        if rate is not None and rate > 0.0:
            failures.append(f"{label} occurred: {tag} = {rate:.4f}, required 0")

    visible = value_of("Entry/target_visible_rate")
    if visible is not None and visible < min_target_visible_rate:
        failures.append(f"the selected opening was rarely visible: {visible:.3f} < {min_target_visible_rate}")

    crossing_time = value_of("Entry/crossing_time_s")
    if crossing_time is not None and crossing_time > max_crossing_time_s:
        failures.append(f"crossing took too long: {crossing_time:.2f} s > {max_crossing_time_s} s")

    return failures


def load_latest_metrics(log_dir: Path) -> dict[str, float]:
    """Read the latest scalar for each tag in a TensorBoard run directory."""

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    metrics: dict[str, float] = {}
    for tag in accumulator.Tags().get("scalars", []):
        values = accumulator.Scalars(tag)
        if values:
            metrics[tag] = float(values[-1].value)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="exploratory")
    parser.add_argument("--min-success-rate", type=float, default=None)
    parser.add_argument("--min-target-visible-rate", type=float, default=None)
    parser.add_argument("--min-clearance-m", type=float, default=0.5)
    parser.add_argument("--max-crossing-time-s", type=float, default=30.0)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    failures = evaluate_metrics(
        load_latest_metrics(args.log_dir),
        min_success_rate=(
            args.min_success_rate if args.min_success_rate is not None else profile["min_success_rate"]
        ),
        min_target_visible_rate=(
            args.min_target_visible_rate
            if args.min_target_visible_rate is not None
            else profile["min_target_visible_rate"]
        ),
        min_clearance_m=args.min_clearance_m,
        max_crossing_time_s=args.max_crossing_time_s,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"PASS: entry skill meets the {args.profile} gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
