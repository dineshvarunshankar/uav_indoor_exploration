#!/usr/bin/env python3
"""Validate one obstacle-navigation result for each simulator environment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


_RECORD_FIELDS = (
    "environment",
    "seed",
    "difficulty",
    "success",
    "collision",
    "timeout",
    "workspace_exit",
    "minimum_clearance_m",
    "altitude_violation_s",
    "stall_time_s",
    "path_efficiency",
    "body_rate_max_rad_s",
    "yaw_acceleration_max_rad_s2",
    "action_saturation_fraction",
)
_LIMIT_RULES = (
    ("minimum_clearance_m", "minimum_clearance_m", "minimum"),
    ("altitude_violation_s", "maximum_altitude_violation_s", "maximum"),
)
_CONTROL_FIELDS = ("clear_contact_force_n", "overlap_contact_force_n", "contact_threshold_n")


@dataclass(frozen=True)
class GateResult:
    """The complete gate result."""

    passed: bool
    failures: tuple[str, ...]


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def assemble_gate_artifact(
    evaluations: list[dict[str, Any]],
    collider_artifact: dict[str, Any],
    limits: dict[str, float],
    contact_threshold_n: float,
) -> dict[str, Any]:
    """Combine held-out records and collider controls without averaging environments."""

    records = [record for evaluation in evaluations for record in evaluation.get("records", [])]
    collider_records = collider_artifact.get("records", [])
    if not collider_records:
        controls: dict[str, float] = {}
    else:
        controls = {
            "clear_contact_force_n": max(
                float(record["clear_contact_force_n"]) for record in collider_records
            ),
            "overlap_contact_force_n": min(
                float(record["overlap_contact_force_n"]) for record in collider_records
            ),
            "contact_threshold_n": float(contact_threshold_n),
        }
    return {"controls": controls, "limits": dict(limits), "records": records}


def validate_gate_artifact(
    artifact: dict[str, Any], required_difficulties: set[int]
) -> GateResult:
    """Apply all limits to each record and retain earlier difficulty gates."""

    failures: list[str] = []
    controls = artifact.get("controls")
    if not isinstance(controls, dict) or any(field not in controls for field in _CONTROL_FIELDS):
        failures.append("physics controls are incomplete")
    else:
        if not all(_finite_number(controls[field]) for field in _CONTROL_FIELDS):
            failures.append("physics controls contain a non-finite value")
        elif not (
            controls["clear_contact_force_n"] <= controls["contact_threshold_n"]
            and controls["overlap_contact_force_n"] > controls["contact_threshold_n"]
        ):
            failures.append("physics controls do not separate clear and overlap contact")

    limits = artifact.get("limits")
    required_limits = {limit for _, limit, _ in _LIMIT_RULES}
    if not isinstance(limits, dict) or not required_limits.issubset(limits):
        failures.append("gate limits are incomplete")
        limits = None
    elif not all(_finite_number(limits[name]) for name in required_limits):
        failures.append("gate limits contain a non-finite value")
        limits = None

    records = artifact.get("records")
    if not isinstance(records, list) or not records:
        failures.append("per-environment records are absent")
        records = []

    found_difficulties: set[int] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(f"record {index} is not an object")
            continue
        missing = [field for field in _RECORD_FIELDS if field not in record]
        if missing:
            failures.append(f"record {index} is missing {', '.join(missing)}")
            continue
        environment = record["environment"]
        difficulty = record["difficulty"]
        if isinstance(difficulty, int) and not isinstance(difficulty, bool):
            found_difficulties.add(difficulty)
        label = f"environment {environment}"
        if record["success"] is not True:
            failures.append(f"{label} did not reach its goal")
        if record["collision"] is not False:
            failures.append(f"{label} collided")
        if record["timeout"] is not False:
            failures.append(f"{label} timed out")
        if record["workspace_exit"] is not False:
            failures.append(f"{label} left its workspace")
        for field in _RECORD_FIELDS:
            if field in {"success", "collision", "timeout", "workspace_exit"}:
                continue
            if not _finite_number(record[field]):
                failures.append(f"{label} has an invalid {field}")
        if limits is not None:
            for field, limit_name, direction in _LIMIT_RULES:
                value = record[field]
                if not _finite_number(value):
                    continue
                limit = limits[limit_name]
                failed = value < limit if direction == "minimum" else value > limit
                if failed:
                    failures.append(f"{label} has {field}={value} outside {limit_name}={limit}")

    for difficulty in sorted(required_difficulties - found_difficulties):
        failures.append(f"difficulty {difficulty} has no per-environment record")
    return GateResult(passed=not failures, failures=tuple(failures))


def _dry_run_artifact() -> dict[str, Any]:
    return {
        "controls": {
            "clear_contact_force_n": 0.0,
            "overlap_contact_force_n": 100.0,
            "contact_threshold_n": 1.0,
        },
        "limits": {
            "minimum_clearance_m": 0.15,
            "maximum_altitude_violation_s": 0.0,
        },
        "records": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", nargs="?", type=Path)
    parser.add_argument("--evaluation", action="append", type=Path, default=[])
    parser.add_argument("--colliders", type=Path)
    parser.add_argument("--limits", type=Path)
    parser.add_argument("--contact-threshold-n", type=float, default=1.0)
    parser.add_argument("--difficulties", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        artifact = _dry_run_artifact()
    elif args.evaluation:
        if args.colliders is None or args.limits is None:
            parser.error("--evaluation needs --colliders and --limits")
        artifact = assemble_gate_artifact(
            [json.loads(path.read_text()) for path in args.evaluation],
            json.loads(args.colliders.read_text()),
            json.loads(args.limits.read_text()),
            args.contact_threshold_n,
        )
    elif args.artifact is not None:
        artifact = json.loads(args.artifact.read_text())
    else:
        parser.error("give an artifact path, evaluation artifacts, or use --dry-run")
    required = {int(value) for value in args.difficulties.split(",") if value}
    result = validate_gate_artifact(artifact, required)
    print(json.dumps({"status": "PASS" if result.passed else "FAIL", "failures": result.failures}, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
