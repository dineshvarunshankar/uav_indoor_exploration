#!/usr/bin/env python3
"""Print a run's diagnostic series, then its last measurable value for each tag.

Reading only the final scalar of a run has produced three wrong conclusions on this task:
a repair was called a regression because the last iteration had promoted to another
curriculum stage, a facade error read zero because nothing in that iteration was
comparable, and a rate read NaN because one iteration's resets held no measurable sample.
The series is the evidence; the summary is only a convenience.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

# Metric groups.
GROUPS = {
    "coverage": [
        "Curriculum/stage",
        "Selection/on_briefed_face_rate",
        "Mission/facade_hint_supplied",
        "Validation/samples",
    ],
    "facade": [
        "Selection/facade_normal_error_deg",
        "Selection/facade_offset_error_m",
        "Selection/selected_truth_gap_m",
        "Mission/selected_position_truth_match_rate",
    ],
    "association": [
        "Association/by_identity",
        "Association/stored_id_stale",
        "Association/refresh_attempted",
        "Association/refresh_found",
        "Association/stored_box_gap",
    ],
    "sensing": [
        "Validation/tracks_fresh_rate",
        "Validation/target_box_available_rate",
        "Validation/evidence_valid_rate",
        "Validation/clear_rate",
        "Validation/blocked_rate",
    ],
    "mission": [
        "Mission/ledger_occupied",
        "Mission/ledger_true_validated",
        "Mission/ledger_true_inspected",
        "Mission/transit_arrived",
    ],
    "safety": [
        "Safety/collision",
        "Safety/recovery_failed",
        "Safety/unauthorized_crossing",
    ],
    "learning": ["Loss/kl", "Loss/learning_rate", "Loss/learning_rate_ratio", "Loss/value"],
}


def read_series(log_dir: Path) -> dict[str, dict[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: {point.step: float(point.value) for point in accumulator.Scalars(tag)}
        for tag in accumulator.Tags().get("scalars", [])
    }


def print_group(name: str, tags: list[str], series: dict[str, dict[int, float]]) -> None:
    present = [tag for tag in tags if tag in series]
    if not present:
        print(f"\n[{name}] no tags logged")
        return
    steps = sorted({step for tag in present for step in series[tag]})
    print(f"\n[{name}]")
    print("  it " + " ".join(f"{tag.split('/')[-1][:13]:>13s}" for tag in present))
    for step in steps:
        cells = []
        for tag in present:
            value = series[tag].get(step)
            cells.append("            -" if value is None else f"{value:13.3f}")
        print(f"  {step:3d} " + " ".join(cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--group", action="append", default=None, choices=sorted(GROUPS))
    parser.add_argument("--tag", action="append", default=None, help="Extra tag, repeatable.")
    args = parser.parse_args()

    series = read_series(args.log_dir)
    chosen = args.group or sorted(GROUPS)
    for name in chosen:
        print_group(name, GROUPS[name], series)
    if args.tag:
        print_group("requested", args.tag, series)

    print("\n[last measurable value, and the iteration it came from]")
    for name in chosen:
        for tag in GROUPS[name]:
            points = series.get(tag)
            if not points:
                continue
            finite = [(step, value) for step, value in sorted(points.items()) if math.isfinite(value)]
            if not finite:
                print(f"  {tag:46s} never measurable")
                continue
            step, value = finite[-1]
            last_step = max(points)
            stale = "" if step == last_step else f"  (the point at {last_step} was not measurable)"
            print(f"  {tag:46s} {value:12.4f}  at iteration {step}{stale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
