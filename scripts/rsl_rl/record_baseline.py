#!/usr/bin/env python3
"""Freeze a baseline: run path, source state, seed, checkpoint, and metric table.

A later policy is only comparable to a baseline that says what code produced it. This
writes one Markdown record beside the run, so a comparison never rests on remembered
numbers or on a run whose gates have since changed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def source_state() -> dict[str, str]:
    """Return the working-tree revision and whether it carries uncommitted changes."""

    def run(*command: str) -> str:
        try:
            return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unavailable"

    revision = run("git", "rev-parse", "HEAD")
    dirty = run("git", "status", "--porcelain")
    return {
        "revision": revision,
        "uncommitted_changes": "yes" if dirty and dirty != "unavailable" else "no",
        "changed_files": str(len(dirty.splitlines())) if dirty != "unavailable" else "unavailable",
    }


def latest_checkpoint(log_dir: Path) -> str:
    """Return the highest-numbered checkpoint in the run directory."""

    checkpoints = sorted(log_dir.glob("model_*.pt"), key=lambda path: path.stat().st_mtime)
    return checkpoints[-1].name if checkpoints else "none"


def metric_table(metrics: dict[str, float], prefixes: tuple[str, ...]) -> str:
    """Render the gated metrics as a Markdown table, grouped by prefix."""

    lines = ["| metric | value |", "| --- | --- |"]
    for prefix in prefixes:
        for tag in sorted(tag for tag in metrics if tag.startswith(prefix)):
            lines.append(f"| {tag} | {metrics[tag]:.6g} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", type=str, required=True, help="What this baseline is for.")
    parser.add_argument(
        "--prefix",
        action="append",
        default=None,
        help="Metric prefixes to record; repeatable. Defaults to the gated groups.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from check_building_inspection_run import load_latest_metrics  # noqa: PLC0415

    prefixes = tuple(args.prefix) if args.prefix else (
        "Mission/",
        "Validation/",
        "Association/",
        "Safety/",
        "Perception/",
        "Curriculum/",
        "Loss/",
        "Train/",
        "Episode_Termination/",
    )
    metrics = load_latest_metrics(args.log_dir)
    state = source_state()
    output = args.output or args.log_dir / "baseline_record.md"
    output.write_text(
        f"# Baseline: {args.label}\n\n"
        f"- recorded: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
        f"- run path: `{args.log_dir}`\n"
        f"- seed: {args.seed}\n"
        f"- checkpoint: `{latest_checkpoint(args.log_dir)}`\n"
        f"- source revision: `{state['revision']}`\n"
        f"- uncommitted changes: {state['uncommitted_changes']} ({state['changed_files']} files)\n\n"
        "## Final metrics\n\n" + metric_table(metrics, prefixes) + "\n",
        encoding="utf-8",
    )
    (args.log_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
