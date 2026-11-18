#!/usr/bin/env python3
"""Qualify transit behavior or the nominal rough-proposal search integration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
from pathlib import Path

from gate_json import dumps_strict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Uav-Building-Inspection-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--min_success_rate", type=float, default=0.9)
parser.add_argument("--max_collision_rate", type=float, default=0.0)
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--proposal_file", type=str, default=None)
parser.add_argument("--proposal_map_version", type=int, default=0)
parser.add_argument("--simulation_association_residual_m", type=float, default=None)
parser.add_argument("--proposal_search_yaw_half_span_rad", type=float, default=None)
parser.add_argument("--proposal_search_yaw_rate_rad_s", type=float, default=None)
parser.add_argument("--output_json", "--result", dest="output_json", type=str, default=None)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import uav_indoor.tasks  # noqa: F401
from rsl_rl.runners import OnPolicyRunner
from uav_indoor.tasks.manager_based.uav_indoor.building_inspection.proposals import (
    validate_association_residual,
)
from uav_indoor.tasks.manager_based.uav_indoor.inspection import MissionStage

from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def _write_json(result: dict) -> None:
    text = dumps_strict(result, indent=2, sort_keys=True)
    if args_cli.output_json is None:
        print(text)
    else:
        Path(args_cli.output_json).write_text(text + "\n", encoding="utf-8")


def _proposal_record_errors(record: dict) -> list[str]:
    errors: list[str] = []
    search = record["proposal_search_first_step"]
    ledger = record["proposal_ledger_first_step"]
    matched = record["proposal_match_first_step"]
    if search < 0:
        errors.append("hint never entered SEARCHING")
    if ledger < 0:
        errors.append("no detector-backed ledger entry appeared")
    if matched < 0:
        errors.append("no proposal reached MATCHED")
    if matched >= 0 and search >= 0 and matched < search:
        errors.append("matching preceded search activation")
    if matched >= 0 and ledger >= 0 and matched < ledger:
        errors.append("matching preceded detector evidence")
    if record["proposal_direct_reference"]:
        errors.append("a rough proposal became an APPROACH reference")
    if record["unresolved_proposals"] > 0 and record["mission_success"]:
        errors.append("mission reported success with unresolved proposals")
    return errors


def main() -> int:
    """Run a bounded deterministic gate and report measured proposal state."""

    proposal_mode = args_cli.proposal_file is not None
    if proposal_mode and args_cli.simulation_association_residual_m is None:
        print("FAIL: nominal proposal mode requires an explicit simulation association residual")
        return 2
    if proposal_mode and (
        args_cli.proposal_search_yaw_half_span_rad is None
        or args_cli.proposal_search_yaw_rate_rad_s is None
    ):
        print("FAIL: nominal proposal mode requires an explicit search yaw half-span and rate")
        return 2
    residual = validate_association_residual(args_cli.simulation_association_residual_m)
    if args_cli.seed < 0:
        print("FAIL: seed must be non-negative")
        return 2
    if args_cli.checkpoint is not None and not proposal_mode:
        print("FAIL: checkpoint mode is only defined for the proposal integration gate")
        return 2
    checkpoint = args_cli.checkpoint.expanduser().resolve() if args_cli.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        print(f"FAIL: checkpoint does not exist: {checkpoint}")
        return 2
    checkpoint_sha256 = None
    if checkpoint is not None:
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    if proposal_mode:
        env_cfg.opening_proposal_path = args_cli.proposal_file
        env_cfg.proposal_map_version = args_cli.proposal_map_version
        env_cfg.maximum_association_residual_m = residual
        env_cfg.mission.proposal_search_yaw_half_span_rad = args_cli.proposal_search_yaw_half_span_rad
        env_cfg.mission.proposal_search_yaw_rate_rad_s = args_cli.proposal_search_yaw_rate_rad_s
        env_cfg.mission.initial_stage = int(MissionStage.FULL)
        env_cfg.mission.lock_stage = True
        env_cfg.mission.max_floors = 1
        env_cfg.mission.facades_per_floor = 1
        env_cfg.mission.blank_stations_before_turn = 1
        env_cfg.episode_length_s = 120.0
        env_cfg.record_episode_results = True

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    task = gym_env.unwrapped
    runner = None
    policy = None
    observations = None
    if checkpoint is not None:
        installed_version = metadata.version("rsl-rl-lib")
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        agent_cfg.device = args_cli.device or "cuda:0"
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=task.device)
        observations = env.get_observations()
    else:
        env = gym_env
        env.reset()
    first_record = len(task.episode_records)

    finished = 0
    transit_completions = 0
    collisions = 0
    timeouts = 0
    try:
        while simulation_app.is_running() and finished < args_cli.episodes:
            with torch.inference_mode():
                if policy is not None:
                    action = policy(observations)
                    observations, _, done, _ = env.step(action)
                    if hasattr(policy, "reset"):
                        policy.reset(done)
                    elif runner is not None:
                        runner.alg.actor.reset(done)
                    done_ids = done.nonzero(as_tuple=False).flatten().tolist()
                else:
                    action_term = task.action_manager.get_term("px4_waypoint")
                    robot_position = task.scene["robot"].data.root_pos_w - task.scene.env_origins
                    robot_quaternion = task.scene["robot"].data.root_quat_w
                    reference = task.mission_reference
                    delta_body = quat_apply_inverse(
                        robot_quaternion,
                        reference.position_w - robot_position,
                    )
                    reach = torch.tensor(
                        (
                            action_term.cfg.v_max_xy,
                            action_term.cfg.v_max_xy,
                            action_term.cfg.v_max_z,
                        ),
                        device=task.device,
                    ) * action_term.cfg.reach_seconds
                    action = torch.empty(env.action_space.shape, device=task.device)
                    action[:, :3] = delta_body / reach
                    yaw_reach = action_term.cfg.yaw_rate_max * action_term.cfg.reach_seconds
                    action[:, 3] = wrap_to_pi(reference.yaw_w - task.robot_yaw_w) / yaw_reach
                    action.clamp_(-1.0, 1.0)
                    _, _, terminated, truncated, _ = env.step(action)
                    mission_done = task.termination_manager.get_term("mission_complete")
                    collision = task.termination_manager.get_term("collision")
                    done = terminated | truncated
                    done_ids = done.nonzero(as_tuple=False).flatten().tolist()
                    for env_id in done_ids:
                        transit_completions += int(mission_done[env_id].item())
                        collisions += int(collision[env_id].item())
                        timeouts += int(truncated[env_id].item())
                finished += min(len(done_ids), args_cli.episodes - finished)
    finally:
        env.close()

    denominator = max(finished, 1)
    collision_rate = collisions / denominator
    if not proposal_mode:
        success_rate = transit_completions / denominator
        failures: list[str] = []
        if finished < args_cli.episodes:
            failures.append("simulator stopped before the requested episodes completed")
        if success_rate < args_cli.min_success_rate:
            failures.append(f"success rate {success_rate:.3f} < {args_cli.min_success_rate:.3f}")
        if collision_rate > args_cli.max_collision_rate:
            failures.append(
                f"collision rate {collision_rate:.3f} > {args_cli.max_collision_rate:.3f}"
            )
        result = {
            "mode": "transit",
            "verdict": "FAIL" if failures else "PASS",
            "episodes": finished,
            "requested_episodes": args_cli.episodes,
            "successes": transit_completions,
            "collisions": collisions,
            "timeouts": timeouts,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "failures": failures,
        }
        print(
            f"TRANSIT_GATE episodes={finished} successes={transit_completions} collisions={collisions} "
            f"timeouts={timeouts} success_rate={success_rate:.3f} collision_rate={collision_rate:.3f}"
        )
        if args_cli.output_json is not None:
            _write_json(result)
        print(
            "PASS: deterministic local-waypoint transit is ready for short PPO training"
            if not failures
            else "FAIL: transit contract was not observed"
        )
        return int(bool(failures))

    records = [record for record in task.episode_records[first_record:] if record["episode_steps"] > 0]
    records = records[: args_cli.episodes]
    collisions = sum(bool(record["collision"]) for record in records)
    timeouts = sum(bool(record["timeout"]) for record in records)
    denominator = max(len(records), 1)
    collision_rate = collisions / denominator
    record_errors = [_proposal_record_errors(record) for record in records]
    if checkpoint is not None:
        for record, errors in zip(records, record_errors, strict=True):
            phase_steps = record.get("phase_steps")
            if not isinstance(phase_steps, list) or len(phase_steps) <= 3:
                errors.append("phase-step record cannot establish APPROACH and VALIDATE")
            else:
                if phase_steps[2] <= 0:
                    errors.append("policy never entered APPROACH")
                if phase_steps[3] <= 0 and record.get("validation_samples", 0) <= 0:
                    errors.append("policy did not reach VALIDATE")
            if record.get("unauthorized_crossing"):
                errors.append("policy made an unauthorized crossing")
            if record.get("recovery_failed"):
                errors.append("policy recovery failed")
    failures = [error for episode_errors in record_errors for error in episode_errors]
    if len(records) < args_cli.episodes:
        failures.append("missing terminal proposal records")
    if collision_rate > args_cli.max_collision_rate:
        failures.append(
            f"collision rate {collision_rate:.3f} > {args_cli.max_collision_rate:.3f}"
        )
    result = {
        "mode": "model109_policy_proposal_transit" if checkpoint is not None else "nominal_proposal_search",
        "controller": "rsl_rl_policy" if checkpoint is not None else "direct_waypoint_oracle",
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "checkpoint_sha256": checkpoint_sha256,
        "verdict": "FAIL" if failures else "PASS",
        "episodes": len(records),
        "requested_episodes": args_cli.episodes,
        "simulation_setting": {
            "association_residual_m": residual,
            "proposal_search_yaw_half_span_rad": args_cli.proposal_search_yaw_half_span_rad,
            "proposal_search_yaw_rate_rad_s": args_cli.proposal_search_yaw_rate_rad_s,
            "map_version": args_cli.proposal_map_version,
            "proposal_file": args_cli.proposal_file,
            "max_floors": 1,
            "facades_per_floor": 1,
            "blank_stations_before_turn": 1,
            "episode_length_s": 120.0,
            "seed": args_cli.seed,
        },
        "collision_rate": collision_rate,
        "records": records,
        "errors": record_errors,
        "failures": failures,
    }
    _write_json(result)
    print(
        "PASS: nominal proposal search preserved detector authority and explicit outcomes"
        if not failures
        else "FAIL: nominal proposal contract was not observed"
    )
    return int(bool(failures))


if __name__ == "__main__":
    result = main()
    simulation_app.close()
    raise SystemExit(result)
