#!/usr/bin/env python3
"""Reject a building-inspection training run that did not leave the hover regime."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def evaluate_metrics(
    metrics: dict[str, float],
    minimum_stage: int | None = None,
    require_truth_validation: bool = False,
) -> list[str]:
    """Return actionable failures for the last logged training interval."""

    failures: list[str] = []

    def below(tag: str, minimum: float, label: str) -> None:
        value = metrics.get(tag)
        below_minimum = (
            value is not None and value < minimum and not math.isclose(value, minimum, rel_tol=1.0e-6, abs_tol=1.0e-12)
        )
        if value is None or not math.isfinite(value) or below_minimum:
            failures.append(f"{label}: expected {tag} >= {minimum}, got {value}")

    def above(tag: str, maximum: float, label: str) -> None:
        value = metrics.get(tag)
        if value is None or not math.isfinite(value) or value > maximum:
            failures.append(f"{label}: expected {tag} <= {maximum}, got {value}")

    def present(tag: str, label: str) -> None:
        """Require a diagnostic to exist and be finite, without judging its value."""

        value = metrics.get(tag)
        if value is None or not math.isfinite(value):
            failures.append(f"{label}: expected {tag} to be logged and finite, got {value}")

    # Truth-matched coverage.
    below(
        "Selection/on_briefed_face_rate",
        0.25,
        "too little of the sample was on the briefed facade for truth to score it",
    )
    below("Mission/transit_arrived", 0.25, "transit never completed")
    stage = metrics.get("Curriculum/stage")
    if minimum_stage is not None and (stage is None or not math.isfinite(stage) or stage < minimum_stage):
        failures.append(
            f"curriculum did not reach required stage: expected Curriculum/stage >= {minimum_stage}, got {stage}"
        )
    gate_stage = float(minimum_stage) if minimum_stage is not None else stage
    if gate_stage is None or gate_stage >= 2.0:
        # Detector-backed opening and coverage checks.
        below("Mission/ledger_occupied", 0.5, "the survey recorded no opening")
        below(
            "Mission/ledger_true_validated",
            0.25,
            "no truth-matched opening was confirmed",
        )
        # Validation outcome rates.
        present("Validation/blocked_rate", "the ToF blocked rate was not logged")
        present("Validation/clear_rate", "the ToF clear rate was not logged")
        below(
            "Validation/target_box_available_rate",
            0.05,
            "the ToF never had a target box to sample",
        )
        below("Validation/evidence_valid_rate", 0.05, "no validation step produced valid ToF evidence")
        below(
            "Mission/selected_position_truth_match_rate",
            0.25,
            "the approach was aimed at a phantom opening",
        )
    if require_truth_validation or gate_stage is None or gate_stage >= 3.0:
        below(
            "Mission/ledger_true_validated",
            0.25,
            "no truth-matched opening was ToF-validated",
        )
    if gate_stage is None or gate_stage >= 3.0:
        below("Mission/ledger_true_inspected", 0.25, "no truth-matched opening was inspected")
    above("Episode_Termination/collision", 0.0, "collision safety gate failed")
    above("Episode_Termination/recovery_failed", 0.0, "recovery safety gate failed")
    above("Safety/collision", 0.0, "collision safety gate failed")
    above("Safety/recovery_failed", 0.0, "recovery safety gate failed")
    above("Safety/unauthorized_crossing", 0.0, "the aircraft crossed the facade without authorization")
    below("Loss/learning_rate", 5.0e-5, "learning rate collapsed")

    kl = metrics.get("Loss/kl")
    if kl is None or not math.isfinite(kl) or not 0.0 <= kl <= 0.1:
        failures.append(f"PPO KL is missing or unstable: expected 0 <= Loss/kl <= 0.1, got {kl}")
    return failures


def load_latest_metrics(log_dir: Path) -> dict[str, float]:
    """Read the last finite scalar for each TensorBoard tag."""

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    metrics: dict[str, float] = {}
    for tag in accumulator.Tags().get("scalars", []):
        for point in reversed(accumulator.Scalars(tag)):
            if math.isfinite(point.value):
                metrics[tag] = float(point.value)
                break
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument(
        "--minimum-stage",
        type=int,
        choices=range(4),
        default=None,
        help="Require promotion to at least this curriculum stage (survey completion is 2).",
    )
    parser.add_argument(
        "--require-truth-validation",
        action="store_true",
        help="Require a truth-matched ToF validation, including in a forced stage-2 gate.",
    )
    args = parser.parse_args()

    failures = evaluate_metrics(
        load_latest_metrics(args.log_dir),
        minimum_stage=args.minimum_stage,
        require_truth_validation=args.require_truth_validation,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: completed curriculum stages and PPO learning gates are healthy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
