#!/usr/bin/env python3
"""Qualify a continuous multi-window mission: pace, coverage, repeat visits, and turning.

The isolated entry gate ends at the crossing, and the bounded mission gate asks only that
one sequence completes. Neither can show whether the aircraft moves on. This gate demands
that a single episode inspects one opening, exits through its protected route, and then
commits to a DIFFERENT opening, and it measures what makes that useful for recording:

- survey-to-entry delay, and where a slow one was waiting. A survey stalls either flying to
  its station or standing on it with no eligible candidate; the split says which,
- distinct openings inspected in one episode, which is coverage,
- repeat entries into an opening this episode already entered, which must be zero,
- body yaw rate and its sign reversals, which is what a gimbal-less camera records.

A zero action leaves the deterministic reference controller in charge, so this measures the
mission itself, which is where the survey order, the ledger, and the phase timing live.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Uav-Building-Inspection-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--episodes", type=int, default=16)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
# Multi-opening episode limit.
parser.add_argument("--episode_length_s", type=float, default=150.0)
parser.add_argument("--min_unique_openings", type=int, default=3)
parser.add_argument(
    "--min_multi_window_fraction",
    type=float,
    default=0.75,
    help="Least fraction of episodes that must each inspect the required distinct openings.",
)
parser.add_argument("--max_survey_to_entry_s", type=float, default=25.0)
# Yaw-rate and heading-reversal limits.
parser.add_argument("--max_yaw_rate_mean_deg_s", type=float, default=10.0)
parser.add_argument("--max_yaw_reversals_per_min", type=float, default=2.0)
parser.add_argument("--max_floors", type=int, default=1)
# Facade count.
parser.add_argument("--facades_per_floor", type=int, default=4)
# Survey span.
parser.add_argument("--sweep_half_span_m", type=float, default=30.0)
# Heading damping.
parser.add_argument("--yaw_gain", type=float, default=1.0)
# Waypoint reach by motion mode.
parser.add_argument(
    "--xy_limit_m",
    type=float,
    nargs=5,
    default=None,
    help="Override the five per-mode waypoint reaches, in metres.",
)
parser.add_argument("--progress_every", type=int, default=200, help="Report progress every N steps.")
parser.add_argument(
    "--min_crossing_observability",
    type=float,
    default=0.95,  # keep equal to safety_evidence.DEFAULT_MIN_OBSERVABILITY
    help="Fraction of steps the crossing test must be able to answer before a PASS.",
)
parser.add_argument("--result", type=str, default=None, help="Write the verdict to this JSON file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Rendered sensor inputs.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import uav_indoor.tasks  # noqa: F401
from uav_indoor.tasks.manager_based.uav_indoor.building_inspection.safety_evidence import (
    crossing_observability,
    safety_verdict,
)
from uav_indoor.tasks.manager_based.uav_indoor.inspection import MissionPhase, MissionStage

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def mean_of(values: list[float]) -> float:
    """Mean over the episodes that could answer, NaN when none could.

    Zero is a real answer to "how fast" and must not stand in for "never happened".
    """

    usable = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.fmean(usable) if usable else float("nan")


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.mission.initial_stage = int(MissionStage.FULL)
    env_cfg.mission.lock_stage = True
    env_cfg.mission.facades_per_floor = args_cli.facades_per_floor
    env_cfg.mission.max_floors = args_cli.max_floors
    env_cfg.mission.sweep_half_span_m = args_cli.sweep_half_span_m
    env_cfg.record_episode_results = True
    if args_cli.yaw_gain != 1.0:
        raise SystemExit(
            "--yaw_gain no longer applies: the position action sends the yaw target "
            "to PX4 directly, with no heading-loop damping to set"
        )
    if args_cli.xy_limit_m is not None:
        env_cfg.actions.px4_waypoint.xy_limit_m = tuple(args_cli.xy_limit_m)
    env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    task = env.unwrapped
    env.reset()
    # Discard startup reset records.
    task.episode_records.clear()

    finished = 0
    steps = 0
    started = time.monotonic()
    records: list[dict] = []
    try:
        while simulation_app.is_running() and finished < args_cli.episodes:
            with torch.inference_mode():
                action = torch.zeros(env.action_space.shape, device=task.device)
                env.step(action)
                steps += 1
                if args_cli.progress_every and steps % args_cli.progress_every == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"progress: {steps} steps, {finished}/{args_cli.episodes} episodes, "
                        f"{elapsed:.0f} s elapsed, {steps / max(elapsed, 1e-6):.2f} steps/s",
                        flush=True,
                    )
                while task.episode_records and finished < args_cli.episodes:
                    records.append(task.episode_records.pop(0))
                    finished += 1
    finally:
        env.close()

    step_dt = float(task.step_dt)
    minutes = [max(r["episode_steps"], 1) * step_dt / 60.0 for r in records]

    multi_window = [r for r in records if r["unique_openings_inspected"] >= args_cli.min_unique_openings]
    wrong_opening = sum(1 for r in records if r["crossed_elsewhere"])
    # Per-episode safety metrics.
    unauthorized = sum(1 for r in records if r["unauthorized_crossing"])
    collisions = sum(1 for r in records if r["collision"])
    unrecovered = sum(1 for r in records if r["recovery_failed"])
    repeats = sum(r["repeat_entries"] for r in records)
    repeat_episodes = sum(1 for r in records if r["repeat_entries"] > 0)
    unique_mean = mean_of([float(r["unique_openings_inspected"]) for r in records])
    unique_max = max((r["unique_openings_inspected"] for r in records), default=0)
    entries = sum(r["entry_events"] for r in records)

    delay_mean = mean_of([r["survey_to_entry_mean_s"] for r in records])
    delay_max = max((r["survey_to_entry_max_s"] for r in records), default=0.0)
    # Pooled survey wait times.
    survey_steps = sum(r["survey_steps"] for r in records)
    station_steps = sum(r["survey_station_steps"] for r in records)
    travel_steps = sum(r["survey_travel_steps"] for r in records)
    approach_steps = sum(r["phase_steps"][int(MissionPhase.APPROACH)] for r in records)
    validate_steps = sum(r["phase_steps"][int(MissionPhase.VALIDATE)] for r in records)

    # Validation outcomes.
    v_success = sum(r["validation_success"] for r in records)
    v_rejected = sum(r["validation_rejected"] for r in records)
    v_samples = sum(r["validation_samples"] for r in records)
    v_invalid = sum(r["validation_tof_invalid"] for r in records)
    v_short = sum(r["validation_tof_short"] for r in records)
    v_clearance = sum(r["validation_clearance_low"] for r in records)

    yaw_mean = mean_of([r["yaw_rate_mean_abs"] for r in records])
    yaw_max = max((r["yaw_rate_max_abs"] for r in records), default=0.0)
    reversals_per_min = mean_of(
        [r["yaw_reversals"] / minute for r, minute in zip(records, minutes, strict=True)]
    )
    observe_yaw_mean = mean_of([r["observe_yaw_rate_mean_abs"] for r in records])
    observe_reversals = sum(r["observe_yaw_reversals"] for r in records)

    degrees = 180.0 / math.pi
    print(
        f"MULTI_WINDOW_GATE episodes={finished} entries={entries} "
        f"unique_openings_mean={unique_mean:.2f} unique_openings_max={unique_max} "
        f"multi_window_episodes={len(multi_window)}/{finished} "
        f"repeat_entries={repeats} in {repeat_episodes} episodes wrong_opening={wrong_opening} "
        f"unauthorized={unauthorized} collisions={collisions} recovery_failed={unrecovered} "
        f"survey_to_entry_mean_s={delay_mean:.2f} survey_to_entry_max_s={delay_max:.2f} "
        f"survey_s={survey_steps * step_dt:.0f} (at_station={station_steps * step_dt:.0f} "
        f"travelling={travel_steps * step_dt:.0f}) approach_s={approach_steps * step_dt:.0f} "
        f"validate_s={validate_steps * step_dt:.0f} "
        f"validation_success={v_success} validation_rejected={v_rejected} "
        f"validation_samples={v_samples} (tof_invalid={v_invalid} tof_short={v_short} "
        f"clearance_low={v_clearance}) "
        f"yaw_mean_deg_s={yaw_mean * degrees:.2f} yaw_max_deg_s={yaw_max * degrees:.2f} "
        f"yaw_reversals_per_min={reversals_per_min:.2f} "
        f"observe_yaw_mean_deg_s={observe_yaw_mean * degrees:.2f} "
        f"observe_yaw_reversals={observe_reversals}",
        flush=True,
    )

    failures = []
    if finished < args_cli.episodes:
        failures.append("the simulator stopped before the requested episodes completed")
    fraction = len(multi_window) / max(finished, 1)
    if fraction < args_cli.min_multi_window_fraction:
        failures.append(
            f"only {len(multi_window)} of {finished} episodes ({fraction:.3f}) inspected "
            f"{args_cli.min_unique_openings} distinct openings, needed "
            f"{args_cli.min_multi_window_fraction:.3f}: the sweep is not moving on"
        )
    if wrong_opening:
        failures.append(f"{wrong_opening} crossings of the wrong opening, required 0")
    for tag, value in (
        ("episodes with an unauthorized crossing", unauthorized),
        ("collisions", collisions),
        ("recovery failures", unrecovered),
    ):
        if value:
            failures.append(f"{value} {tag}, required 0")
    if repeats:
        failures.append(
            f"{repeats} entries into an opening the same episode had already entered, in "
            f"{repeat_episodes} episodes, required 0"
        )
    if math.isnan(delay_mean):
        failures.append("no episode reached ENTRY, so survey-to-entry delay is unmeasured")
    elif delay_mean > args_cli.max_survey_to_entry_s:
        failures.append(
            f"survey to entry took {delay_mean:.2f} s on average, over the "
            f"{args_cli.max_survey_to_entry_s:g} s budget; the survey spent "
            f"{station_steps * step_dt:.0f} s at its station and "
            f"{travel_steps * step_dt:.0f} s flying to it"
        )
    if math.isnan(yaw_mean):
        failures.append("yaw rate is unmeasured")
    elif yaw_mean * degrees > args_cli.max_yaw_rate_mean_deg_s:
        failures.append(
            f"mean body yaw rate {yaw_mean * degrees:.2f} deg/s exceeds "
            f"{args_cli.max_yaw_rate_mean_deg_s:g} deg/s: not stable enough to record from"
        )
    if reversals_per_min > args_cli.max_yaw_reversals_per_min:
        failures.append(
            f"{reversals_per_min:.2f} heading reversals per minute exceeds "
            f"{args_cli.max_yaw_reversals_per_min:g}: the aircraft is hunting, not turning"
        )
    for failure in failures:
        print(f"FAIL: {failure}", flush=True)
    verdict, unverified = safety_verdict(failures, records, args_cli.min_crossing_observability)
    for note in unverified:
        print(f"UNVERIFIED: {note}", flush=True)
    failures = list(failures) + unverified
    if not failures:
        print(
            "PASS: the mission inspects distinct openings in sequence, without repeats, "
            "at a recordable turn rate",
            flush=True,
        )

    # Write the verdict before simulator shutdown.
    if args_cli.result:
        Path(args_cli.result).write_text(
            json.dumps(
                {
                    "gate": "multi_window",
                    "verdict": verdict,
                    "crossing_observability": crossing_observability(records),
                    "failures": failures,
                    "task": args_cli.task,
                    "seed": args_cli.seed,
                    "episode_length_s": args_cli.episode_length_s,
                    "requested_episodes": args_cli.episodes,
                    "episodes": finished,
                    "entries": entries,
                    "unique_openings_mean": unique_mean,
                    "unique_openings_max": unique_max,
                    "multi_window_episodes": len(multi_window),
                    "repeat_entries": repeats,
                    "wrong_opening_crossings": wrong_opening,
                    "unauthorized_crossing_episodes": unauthorized,
                    "collisions": collisions,
                    "recovery_failures": unrecovered,
                    "yaw_gain": args_cli.yaw_gain,
                    "xy_limit_m": args_cli.xy_limit_m,
                    "repeat_episodes": repeat_episodes,
                    "survey_to_entry_mean_s": delay_mean,
                    "survey_to_entry_max_s": delay_max,
                    "survey_s": survey_steps * step_dt,
                    "survey_at_station_s": station_steps * step_dt,
                    "survey_travelling_s": travel_steps * step_dt,
                    "approach_s": approach_steps * step_dt,
                    "validate_s": validate_steps * step_dt,
                    "validation_success": v_success,
                    "validation_rejected": v_rejected,
                    "validation_samples": v_samples,
                    "validation_tof_invalid": v_invalid,
                    "validation_tof_short": v_short,
                    "validation_clearance_low": v_clearance,
                    "yaw_rate_mean_deg_s": yaw_mean * degrees,
                    "yaw_rate_max_deg_s": yaw_max * degrees,
                    "yaw_reversals_per_min": reversals_per_min,
                    "observe_yaw_rate_mean_deg_s": observe_yaw_mean * degrees,
                    "observe_yaw_reversals": observe_reversals,
                    "records": records,
                },
                indent=2,
            )
        )
    return 1 if failures else 0


if __name__ == "__main__":
    status = main()
    simulation_app.close()
    raise SystemExit(status)
