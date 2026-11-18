#!/usr/bin/env python3
"""Train and gate one continuous obstacle-navigation checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

TASK = "Template-Uav-Obstacle-Navigation-v0"
STAGE_NAMES = ("open", "sparse", "medium", "dense")
DEFAULT_STEPS_PER_ENVIRONMENT = 256
DEFAULT_TRAINING_LAYOUT_BANK_SIZE = 256
_RECORD_BOOLEAN_FIELDS = ("success", "collision", "workspace_exit", "timeout")
_RECORD_NUMBER_FIELDS = (
    "environment",
    "seed",
    "difficulty",
    "minimum_clearance_m",
    "altitude_violation_s",
    "stall_time_s",
    "path_efficiency",
    "body_rate_max_rad_s",
    "yaw_acceleration_max_rad_s2",
    "action_saturation_fraction",
)


def evaluation_seed_set(
    *, seed_start: int, difficulty: int, episodes: int
) -> tuple[int, ...]:
    """Return the complete seed set before an evaluation starts."""

    first = seed_start + difficulty * episodes
    return tuple(range(first, first + episodes))


def training_seed_start_for_round(
    *,
    seed_start: int,
    difficulty: int,
    round_index: int,
    maximum_rounds: int,
    bank_size: int,
    difficulty_count: int = len(STAGE_NAMES),
) -> int:
    """Return the first seed of a disjoint full-difficulty bank for one round."""

    if round_index < 1 or maximum_rounds < 1 or bank_size < 1:
        raise ValueError("round_index, maximum_rounds, and bank_size must be positive")
    ordinal = difficulty * maximum_rounds + round_index - 1
    return seed_start + ordinal * difficulty_count * bank_size


def resume_arguments(checkpoint: Path) -> list[str]:
    """Return arguments that keep the resolved checkpoint path unchanged."""

    return ["--resume", "--resume_path", str(checkpoint.resolve())]


def rehearsal_allocation(
    difficulty: int, boosts: tuple[float, float, float, float]
) -> dict[int, float]:
    """Return the normalized training allocation for the active stage."""

    if difficulty < 0 or difficulty >= len(STAGE_NAMES):
        raise ValueError("difficulty is outside the curriculum")
    if len(boosts) != len(STAGE_NAMES) or any(value < 0.0 for value in boosts):
        raise ValueError("rehearsal boosts must be four non-negative values")
    if difficulty == 0:
        return {0: 1.0}
    stages = range(difficulty + 1)
    raw = [0.30 / difficulty + boosts[stage] for stage in range(difficulty)]
    raw.append(0.70 + boosts[difficulty])
    total = sum(raw)
    return {stage: raw[stage] / total for stage in stages}


def updated_rehearsal_boosts(
    boosts: tuple[float, float, float, float], *, failed_difficulties: set[int]
) -> tuple[float, float, float, float]:
    """Increase sampling for each stage that failed its exact gate."""

    updated = list(boosts)
    for difficulty in failed_difficulties:
        if difficulty < 0 or difficulty >= len(updated):
            raise ValueError("failed difficulty is outside the curriculum")
        updated[difficulty] = min(updated[difficulty] + 0.10, 1.0)
    return tuple(updated)


def rehearsal_override(boosts: tuple[float, float, float, float]) -> str:
    """Return the Hydra override used by the environment sampler."""

    values = ",".join(f"{value:.1f}" for value in boosts)
    return f"env.rehearsal_boosts=[{values}]"


def rollout_length_override(steps_per_environment: int) -> str:
    """Return the Hydra override that controls the collected rollout length."""

    if steps_per_environment <= 0:
        raise ValueError("steps_per_environment must be positive")
    return f"agent.num_steps_per_env={steps_per_environment}"


def obstacle_count_override(difficulty: int, count: int) -> str:
    """Return a training-only obstacle-count override for one stage."""

    if difficulty < 0 or difficulty >= len(STAGE_NAMES):
        raise ValueError("difficulty is outside the curriculum")
    if count < 0 or count > 20:
        raise ValueError("obstacle count must be between 0 and 20")
    if (difficulty == 0) != (count == 0):
        raise ValueError("OPEN requires zero obstacles; other stages require at least one")
    counts = [0, 6, 12, 20]
    counts[difficulty] = count
    values = ",".join(str(value) for value in counts)
    return f"env.obstacle_count_overrides=[{values}]"


def training_transition_count(
    *, iterations: int, num_envs: int, steps_per_environment: int
) -> int:
    """Return the transition count for the exact rollout configuration."""

    if iterations <= 0 or num_envs <= 0 or steps_per_environment <= 0:
        raise ValueError("iterations, num_envs, and steps_per_environment must be positive")
    return iterations * num_envs * steps_per_environment


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Replace a report only after its complete JSON is on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(run_dir: Path) -> Path:
    """Return the checkpoint with the largest numeric iteration."""

    checkpoints = [path for path in run_dir.glob("model_*.pt") if _checkpoint_iteration(path) >= 0]
    if not checkpoints:
        raise RuntimeError(f"no checkpoint in {run_dir}")
    return max(checkpoints, key=_checkpoint_iteration)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def file_digest(path: Path) -> str:
    """Return the SHA-256 digest for one evidence or configuration file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_collider_evidence(
    artifact: dict[str, Any], *, contact_threshold_n: float
) -> list[str]:
    """Validate each physical clear-contact and overlap control."""

    failures: list[str] = []
    if artifact.get("status") != "PASS":
        failures.append("collider artifact status is not PASS")
    records = artifact.get("records")
    if not isinstance(records, list) or not records:
        return failures + ["collider records are absent"]
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(f"collider record {index} is not an object")
            continue
        family = record.get("family", index)
        clear = record.get("clear_contact_force_n")
        overlap = record.get("overlap_contact_force_n")
        if not _finite_number(clear) or not _finite_number(overlap):
            failures.append(f"collider family {family} has invalid contact evidence")
            continue
        if float(clear) > contact_threshold_n:
            failures.append(f"collider family {family} has contact in its clear control")
        if float(overlap) <= contact_threshold_n:
            failures.append(f"collider family {family} has no overlap contact")
        if record.get("passed") is not True:
            failures.append(f"collider family {family} did not pass")
    return failures


def validate_evaluation(
    artifact: dict[str, Any],
    *,
    difficulty: int,
    expected_seeds: tuple[int, ...],
    expected_checkpoint: Path,
    minimum_clearance_m: float,
) -> list[str]:
    """Return strict per-episode failures for one held-out difficulty."""

    if not _finite_number(minimum_clearance_m) or minimum_clearance_m < 0.0:
        return ["minimum clearance threshold must be a finite non-negative number"]
    failures: list[str] = []
    episodes = len(expected_seeds)
    if artifact.get("status") != "RECORDED":
        return [f"difficulty {difficulty}: evaluator status is {artifact.get('status')!r}"]
    if artifact.get("difficulty") != difficulty:
        failures.append(f"difficulty {difficulty}: artifact has wrong difficulty")
    artifact_checkpoint = artifact.get("checkpoint")
    if not isinstance(artifact_checkpoint, str):
        failures.append(f"difficulty {difficulty}: checkpoint is missing")
    elif Path(artifact_checkpoint).expanduser().resolve() != expected_checkpoint.resolve():
        failures.append(f"difficulty {difficulty}: artifact has wrong checkpoint")
    if artifact.get("requested_episodes") != episodes:
        failures.append(f"difficulty {difficulty}: requested episode count does not match")
    if artifact.get("requested_seeds") != list(expected_seeds):
        failures.append(f"difficulty {difficulty}: requested seed set does not match")

    records = artifact.get("records")
    if not isinstance(records, list) or len(records) != episodes:
        failures.append(f"difficulty {difficulty}: expected {episodes} records")
        return failures

    seeds: set[int] = set()
    for index, record in enumerate(records):
        label = f"difficulty {difficulty} record {index}"
        if not isinstance(record, dict):
            failures.append(f"{label}: record is not an object")
            continue
        for field in (*_RECORD_BOOLEAN_FIELDS, *_RECORD_NUMBER_FIELDS):
            if field not in record:
                failures.append(f"{label}: {field} is missing")
        for field in _RECORD_NUMBER_FIELDS:
            if field in record and not _finite_number(record[field]):
                failures.append(f"{label}: {field} is invalid")
        seed = record.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed in seeds:
            failures.append(f"{label}: seed is missing or duplicated")
        else:
            seeds.add(seed)
        if record.get("difficulty") != difficulty:
            failures.append(f"{label}: wrong difficulty")
        if record.get("success") is not True:
            failures.append(f"{label}: success did not pass")
        for field in ("collision", "workspace_exit", "timeout"):
            if record.get(field) is not False:
                failures.append(f"{label}: {field} did not pass")
        clearance = record.get("minimum_clearance_m")
        if not _finite_number(clearance):
            failures.append(f"{label}: minimum clearance is invalid")
        elif float(clearance) < minimum_clearance_m:
            failures.append(
                f"{label}: clearance {float(clearance):.3f} m is below "
                f"{minimum_clearance_m:.3f} m"
            )
        altitude = record.get("altitude_violation_s")
        if not _finite_number(altitude):
            failures.append(f"{label}: altitude violation is invalid")
        elif float(altitude) != 0.0:
            failures.append(f"{label}: altitude violation is {float(altitude):.3f} s")
    if seeds != set(expected_seeds):
        failures.append(f"difficulty {difficulty}: record seed set does not match")
    return failures


def _run(command: list[str], cwd: Path, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def _run_recorded(
    command: list[str],
    cwd: Path,
    dry_run: bool,
    *,
    report: dict[str, Any],
    artifact_path: Path,
) -> None:
    """Run one child process and record an interruption or error."""

    try:
        _run(command, cwd, dry_run)
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
        report["error"] = "KeyboardInterrupt"
        write_report(artifact_path, report)
        raise
    except Exception as error:
        report["status"] = "ERROR"
        report["error"] = repr(error)
        write_report(artifact_path, report)
        raise


def _new_run(log_root: Path, previous: set[Path]) -> Path:
    current = {path for path in log_root.iterdir() if path.is_dir()}
    created = current - previous
    if len(created) != 1:
        raise RuntimeError(f"expected one new run in {log_root}, found {sorted(created)}")
    return created.pop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python"),
    )
    parser.add_argument("--experiment", default="uav_full_mission")
    parser.add_argument("--start-difficulty", type=int, choices=range(4), default=0)
    parser.add_argument("--stop-difficulty", type=int, choices=range(4), default=3)
    parser.add_argument("--iterations-per-round", type=int, default=10)
    parser.add_argument("--max-rounds-per-stage", type=int, default=30)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--evaluation-episodes", type=int, default=32)
    parser.add_argument("--evaluation-seed-start", type=int, default=1_000_000)
    parser.add_argument("--training-seed-start", type=int, default=10_000)
    parser.add_argument(
        "--training-layout-bank-size",
        type=int,
        default=DEFAULT_TRAINING_LAYOUT_BANK_SIZE,
    )
    parser.add_argument(
        "--steps-per-environment", type=int, default=DEFAULT_STEPS_PER_ENVIRONMENT
    )
    parser.add_argument("--training-obstacle-count", type=int)
    parser.add_argument(
        "--collider-artifact",
        type=Path,
        default=Path("logs/gates/2026-09-10_obstacle_navigation_colliders.json"),
    )
    parser.add_argument("--contact-threshold-n", type=float, default=1.0)
    parser.add_argument("--minimum-clearance-m", type=float, default=0.15)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("logs/gates/obstacle_navigation_curriculum.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.start_difficulty > args.stop_difficulty:
        parser.error("--start-difficulty must not exceed --stop-difficulty")
    positive = (
        args.iterations_per_round,
        args.max_rounds_per_stage,
        args.num_envs,
        args.evaluation_episodes,
        args.steps_per_environment,
        args.training_layout_bank_size,
    )
    if any(value <= 0 for value in positive):
        parser.error("iteration, round, environment, and episode counts must be positive")
    if not _finite_number(args.minimum_clearance_m) or args.minimum_clearance_m < 0.0:
        parser.error("--minimum-clearance-m must be a finite non-negative number")
    if args.evaluation_episodes < 32:
        parser.error("--evaluation-episodes must be at least 32 for promotion")
    if args.num_envs < args.evaluation_episodes:
        parser.error("--num-envs must be at least --evaluation-episodes for fixed seed evaluation")
    if args.evaluation_seed_start < 0 or args.training_seed_start < 0:
        parser.error("training and evaluation seed starts must be non-negative")
    if not _finite_number(args.contact_threshold_n) or args.contact_threshold_n < 0.0:
        parser.error("--contact-threshold-n must be a finite non-negative number")
    if args.training_obstacle_count is not None:
        try:
            obstacle_count_override(args.start_difficulty, args.training_obstacle_count)
        except ValueError as error:
            parser.error(str(error))
        if args.start_difficulty != args.stop_difficulty:
            parser.error("--training-obstacle-count requires one active difficulty")

    repo = Path(__file__).resolve().parents[2]
    log_root = repo / "logs" / "rsl_rl" / args.experiment
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")
    if args.start_difficulty > 0 and checkpoint is None:
        parser.error("a checkpoint is required when starting after OPEN")
    collider_path = (
        (repo / args.collider_artifact).resolve()
        if not args.collider_artifact.is_absolute()
        else args.collider_artifact.resolve()
    )
    if not collider_path.is_file():
        parser.error(f"collider artifact does not exist: {collider_path}")
    collider_artifact = json.loads(collider_path.read_text())
    collider_failures = validate_collider_evidence(
        collider_artifact, contact_threshold_n=args.contact_threshold_n
    )
    if collider_failures:
        parser.error("; ".join(collider_failures))

    source_paths = (
        Path(__file__).resolve(),
        repo / "scripts/rsl_rl/evaluate_obstacle_navigation.py",
        repo / "scripts/rsl_rl/train.py",
        repo / "scripts/rsl_rl/cli_args.py",
        repo / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/obstacle_navigation/action_observation.py",
        repo / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/obstacle_navigation/contact_history.py",
        repo / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/obstacle_navigation/curriculum.py",
        repo / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/obstacle_navigation/env.py",
        repo / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/obstacle_navigation/env_cfg.py",
        repo / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/obstacle_navigation/layouts.py",
        repo / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/obstacle_navigation/mdp.py",
        repo / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/obstacle_navigation/policy.py",
        repo / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/obstacle_navigation/reward_features.py",
        repo / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/obstacle_navigation/scene.py",
        repo / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/building_inspection/agents/depth_gru.py",
        repo / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/building_inspection/agents/diagnostic_ppo.py",
        repo / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/building_inspection/agents/rsl_rl_ppo_cfg.py",
    )
    source_digests = {
        str(source.relative_to(repo)): file_digest(source) for source in source_paths
    }

    report: dict[str, Any] = {
        "status": "RUNNING",
        "experiment": args.experiment,
        "continuous_checkpoint": True,
        "initial_checkpoint": str(checkpoint) if checkpoint is not None else None,
        "latest_checkpoint": str(checkpoint) if checkpoint is not None else None,
        "cumulative_transitions": 0,
        "qualification": {
            "episodes": args.evaluation_episodes,
            "seed_start": args.evaluation_seed_start,
            "minimum_clearance_m": args.minimum_clearance_m,
        },
        "collider_evidence": {
            "path": str(collider_path),
            "sha256": file_digest(collider_path),
            "contact_threshold_n": args.contact_threshold_n,
        },
        "source_sha256": source_digests,
        "stages": [],
    }
    artifact_path = (repo / args.artifact).resolve() if not args.artifact.is_absolute() else args.artifact
    if not args.dry_run:
        write_report(artifact_path, report)

    rehearsal_boosts = (0.0, 0.0, 0.0, 0.0)

    for difficulty in range(args.start_difficulty, args.stop_difficulty + 1):
        stage: dict[str, Any] = {
            "difficulty": difficulty,
            "name": STAGE_NAMES[difficulty],
            "rounds": [],
            "passed": False,
        }
        report["stages"].append(stage)
        for round_index in range(1, args.max_rounds_per_stage + 1):
            before = {path for path in log_root.iterdir() if path.is_dir()}
            run_name = f"obstacle_{STAGE_NAMES[difficulty]}_round_{round_index:02d}"
            train = [
                str(args.python),
                "scripts/rsl_rl/train.py",
                "--task",
                TASK,
                "--headless",
                "--enable_cameras",
                "--device",
                "cuda:0",
                "--num_envs",
                str(args.num_envs),
                "--max_iterations",
                str(args.iterations_per_round),
                "--save_interval",
                str(args.iterations_per_round),
                "--experiment_name",
                args.experiment,
                "--run_name",
                run_name,
                f"env.initial_difficulty={difficulty}",
            ]
            round_training_seed_start = training_seed_start_for_round(
                seed_start=args.training_seed_start,
                difficulty=difficulty,
                round_index=round_index,
                maximum_rounds=args.max_rounds_per_stage,
                bank_size=args.training_layout_bank_size,
            )
            train.extend(
                [
                    f"env.training_seed_start={round_training_seed_start}",
                    f"env.layout_bank_size={args.training_layout_bank_size}",
                    rehearsal_override(rehearsal_boosts),
                    rollout_length_override(args.steps_per_environment),
                ]
            )
            if args.training_obstacle_count is not None:
                train.append(obstacle_count_override(difficulty, args.training_obstacle_count))
            parent_checkpoint = checkpoint
            if checkpoint is not None:
                train.extend(resume_arguments(checkpoint))
            report["active"] = {
                "difficulty": difficulty,
                "round": round_index,
                "operation": "training",
                "parent_checkpoint": (
                    str(parent_checkpoint) if parent_checkpoint is not None else None
                ),
            }
            if not args.dry_run:
                write_report(artifact_path, report)
            _run_recorded(
                train,
                repo,
                args.dry_run,
                report=report,
                artifact_path=artifact_path,
            )
            if args.dry_run:
                break
            run_dir = _new_run(log_root, before)
            checkpoint = latest_checkpoint(run_dir)
            transitions = training_transition_count(
                iterations=args.iterations_per_round,
                num_envs=args.num_envs,
                steps_per_environment=args.steps_per_environment,
            )
            report["cumulative_transitions"] += transitions

            round_report: dict[str, Any] = {
                "round": round_index,
                "parent_checkpoint": (
                    str(parent_checkpoint) if parent_checkpoint is not None else None
                ),
                "checkpoint": str(checkpoint),
                "training_transitions": transitions,
                "training_layout_seed_start": round_training_seed_start,
                "training_layout_bank_size": args.training_layout_bank_size,
                "training_obstacle_count": args.training_obstacle_count,
                "cumulative_transitions": report["cumulative_transitions"],
                "rehearsal_boosts": list(rehearsal_boosts),
                "rehearsal_allocation": rehearsal_allocation(
                    difficulty, rehearsal_boosts
                ),
                "evaluations": [],
            }
            stage["rounds"].append(round_report)
            failures: list[str] = []
            failed_difficulties: set[int] = set()
            for evaluation_difficulty in range(difficulty + 1):
                output = repo / "logs" / "gates" / f"{run_dir.name}_difficulty_{evaluation_difficulty}.json"
                evaluate = [
                    str(args.python),
                    "scripts/rsl_rl/evaluate_obstacle_navigation.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--difficulty",
                    str(evaluation_difficulty),
                    "--episodes",
                    str(args.evaluation_episodes),
                    "--seed-start",
                    str(args.evaluation_seed_start),
                    "--num_envs",
                    str(min(args.evaluation_episodes, args.num_envs)),
                    "--output",
                    str(output),
                    "--headless",
                    "--device",
                    "cuda:0",
                ]
                report["active"] = {
                    "difficulty": difficulty,
                    "round": round_index,
                    "operation": "evaluation",
                    "evaluation_difficulty": evaluation_difficulty,
                    "checkpoint": str(checkpoint),
                }
                write_report(artifact_path, report)
                _run_recorded(
                    evaluate,
                    repo,
                    False,
                    report=report,
                    artifact_path=artifact_path,
                )
                evaluation = json.loads(output.read_text())
                expected_seeds = evaluation_seed_set(
                    seed_start=args.evaluation_seed_start,
                    difficulty=evaluation_difficulty,
                    episodes=args.evaluation_episodes,
                )
                evaluation_failures = validate_evaluation(
                    evaluation,
                    difficulty=evaluation_difficulty,
                    expected_seeds=expected_seeds,
                    expected_checkpoint=checkpoint,
                    minimum_clearance_m=args.minimum_clearance_m,
                )
                round_report["evaluations"].append(
                    {
                        "difficulty": evaluation_difficulty,
                        "artifact": str(output),
                        "passed": not evaluation_failures,
                        "failures": evaluation_failures,
                    }
                )
                failures.extend(evaluation_failures)
                if evaluation_failures:
                    failed_difficulties.add(evaluation_difficulty)

            if failed_difficulties:
                rehearsal_boosts = updated_rehearsal_boosts(
                    rehearsal_boosts,
                    failed_difficulties=failed_difficulties,
                )
            round_report["next_rehearsal_boosts"] = list(rehearsal_boosts)
            round_report["passed"] = not failures
            round_report["failure_count"] = len(failures)
            report["latest_checkpoint"] = str(checkpoint)
            write_report(artifact_path, report)
            if not failures:
                stage["passed"] = True
                break
        if args.dry_run:
            break
        if not stage["passed"]:
            report["status"] = "BLOCKED"
            write_report(artifact_path, report)
            return 2

    if args.dry_run:
        return 0
    report["status"] = "PASS"
    report.pop("active", None)
    write_report(artifact_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
