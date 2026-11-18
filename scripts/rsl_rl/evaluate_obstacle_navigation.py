#!/usr/bin/env python3
"""Evaluate one checkpoint on one held-out obstacle difficulty."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--difficulty", type=int, choices=range(4), required=True)
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--seed-start", type=int, default=1_000_000)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--trace-stride", type=int, default=0)
parser.add_argument("--disparity-ablation", choices=("normal", "blank", "shuffled"), default="normal")
parser.add_argument("--measure-depth-alignment", action="store_true")
parser.add_argument("--image-source", choices=("midas", "rendered_inverse_depth"), default="midas")
parser.add_argument("--obstacle-count", type=int)
AppLauncher.add_app_launcher_args(parser)
def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write strict JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


args, hydra_args = parser.parse_known_args()
args.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args).app
exit_code = 0

try:
    import importlib.metadata as metadata

    import gymnasium as gym
    import torch
    import uav_indoor.tasks  # noqa: F401
    from rsl_rl.runners import OnPolicyRunner
    from uav_indoor.tasks.manager_based.uav_indoor.perception.disparity import (
        ablate_disparity,
        inverse_depth_alignment,
    )

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    task_name = "Template-Uav-Obstacle-Navigation-v0"
    if args.episodes <= 0 or args.num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")
    if args.num_envs < args.episodes:
        raise ValueError("num_envs must be at least episodes for a fixed seed evaluation")
    if args.seed_start < 0:
        raise ValueError("seed-start must be non-negative")
    if args.obstacle_count is not None:
        if args.obstacle_count < 0 or args.obstacle_count > 20:
            raise ValueError("obstacle-count must be between 0 and 20")
        if (args.difficulty == 0) != (args.obstacle_count == 0):
            raise ValueError("OPEN requires zero obstacles; other stages require at least one")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = args.device or "cuda:0"
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=args.num_envs)
    env_cfg.disparity.source = args.image_source
    env_cfg.use_held_out_layouts = True
    env_cfg.evaluation_difficulty = args.difficulty
    env_cfg.record_episode_records = True
    env_cfg.held_out_seed_start = args.seed_start
    env_cfg.layout_bank_size = args.episodes
    if args.obstacle_count is not None:
        obstacle_counts = list(env_cfg.obstacle_count_overrides)
        obstacle_counts[args.difficulty] = args.obstacle_count
        env_cfg.obstacle_count_overrides = obstacle_counts
    env_cfg.seed = 0
    if args.measure_depth_alignment:
        env_cfg.scene.rgb_camera.data_types = ["rgb", "distance_to_image_plane"]
    first_seed = args.seed_start + args.difficulty * args.episodes
    requested_seeds = tuple(range(first_seed, first_seed + args.episodes))

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.device = device

    gym_env = gym.make(task_name, cfg=env_cfg)
    task = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=device)

    observations = env.get_observations()
    disparity_permutation = None
    if args.disparity_ablation == "shuffled":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        pixels = observations["disparity"].shape[-2] * observations["disparity"].shape[-1]
        disparity_permutation = torch.randperm(pixels, generator=generator).to(device)
    trace_environment = {
        int(task.layout_seed[environment].item()): environment
        for environment in range(min(args.episodes, task.num_envs))
    }
    steps = 0
    record_cursor = 0
    unique_records: dict[int, dict[str, object]] = {}
    traces: dict[int, list[dict[str, object]]] = {seed: [] for seed in requested_seeds}
    depth_alignment_samples: list[torch.Tensor] = []
    maximum_steps = ((args.episodes + args.num_envs - 1) // args.num_envs + 2) * task.max_episode_length
    while len(unique_records) < args.episodes and steps < maximum_steps:
        with torch.inference_mode():
            actor_observations = dict(observations)
            actor_observations["disparity"] = ablate_disparity(
                observations["disparity"], args.disparity_ablation, disparity_permutation
            )
            if args.measure_depth_alignment:
                depth = task.scene["rgb_camera"].data.output["distance_to_image_plane"]
                if depth.ndim == 4:
                    depth = depth[..., 0]
                inverse_depth = torch.where(
                    torch.isfinite(depth) & (depth > 1.0e-6),
                    depth.reciprocal(),
                    torch.zeros_like(depth),
                )
                pooled_inverse = torch.nn.functional.adaptive_max_pool2d(
                    inverse_depth.unsqueeze(1), actor_observations["disparity"].shape[-2:]
                )
                pooled_depth = torch.where(
                    pooled_inverse > 0.0,
                    pooled_inverse.reciprocal(),
                    torch.full_like(pooled_inverse, torch.inf),
                )
                alignment, alignment_valid = inverse_depth_alignment(
                    actor_observations["disparity"], pooled_depth
                )
                depth_alignment_samples.append(alignment[alignment_valid].cpu())
            actions = policy(actor_observations)
            pre_seeds = task.layout_seed.clone()
            pre_position = (
                task.scene["robot"].data.root_pos_w - task.scene.env_origins
            ).clone()
            pre_velocity = task.scene["robot"].data.root_lin_vel_w.clone()
            pre_goal = task.goal_w.clone()
            observations, _, dones, _ = env.step(actions)
            if args.trace_stride > 0 and steps % args.trace_stride == 0:
                target = task.action_manager.get_term("px4_waypoint").target_w
                for seed, environment in trace_environment.items():
                    if (
                        int(pre_seeds[environment].item()) != seed
                        or bool(dones[environment].item())
                    ):
                        continue
                    traces[seed].append(
                        {
                            "step": steps,
                            "position_m": [float(v) for v in pre_position[environment].tolist()],
                            "goal_m": [float(v) for v in pre_goal[environment].tolist()],
                            "goal_distance_m": float(
                                torch.linalg.vector_norm(
                                    pre_goal[environment] - pre_position[environment]
                                ).item()
                            ),
                            "velocity_m_s": [float(v) for v in pre_velocity[environment].tolist()],
                            "altitude_m": float(pre_position[environment, 2].item()),
                            "policy_action": [float(v) for v in actions[environment].tolist()],
                            "position_setpoint_m": [float(v) for v in target[environment].tolist()],
                            "disparity_min": float(actor_observations["disparity"][environment].min().item()),
                            "disparity_max": float(actor_observations["disparity"][environment].max().item()),
                            "disparity_mean": float(actor_observations["disparity"][environment].mean().item()),
                            "disparity_std": float(actor_observations["disparity"][environment].std().item()),
                        }
                    )
            if hasattr(policy, "reset"):
                policy.reset(dones)
            else:
                runner.alg.actor.reset(dones)
        for record in task.completed_records[record_cursor:]:
            seed = int(record["seed"])
            if seed in requested_seeds:
                unique_records.setdefault(seed, record)
        record_cursor = len(task.completed_records)
        steps += 1

    records = [unique_records[seed] for seed in requested_seeds if seed in unique_records]
    route_family_counts: dict[str, int] = {}
    for record in records:
        family = str(record.get("route_family", "unclassified"))
        route_family_counts[family] = route_family_counts.get(family, 0) + 1
    status = "RECORDED" if len(records) == args.episodes else "INCOMPLETE"
    if status != "RECORDED":
        exit_code = 2
    if depth_alignment_samples:
        alignment_values = torch.cat(depth_alignment_samples)
        depth_alignment = {
            "sample_count": int(alignment_values.numel()),
            "mean_pearson_r": float(alignment_values.mean().item()),
            "median_pearson_r": float(alignment_values.median().item()),
            "positive_fraction": float((alignment_values > 0.0).float().mean().item()),
        }
    else:
        depth_alignment = {"sample_count": 0}
    write_json(
        args.output,
            {
                "status": status,
                "checkpoint": str(checkpoint),
                "difficulty": args.difficulty,
                "obstacle_count": args.obstacle_count,
                "requested_episodes": args.episodes,
                "requested_seeds": list(requested_seeds),
                "steps": steps,
                "route_family_counts": route_family_counts,
                "trace_stride": args.trace_stride,
                "disparity_ablation": args.disparity_ablation,
                "image_source": args.image_source,
                "depth_alignment": depth_alignment,
                "traces": {str(seed): values for seed, values in traces.items()}
                if args.trace_stride > 0
                else {},
                "records": records,
            },
    )
    env.close()
except Exception as error:
    exit_code = 1
    write_json(
        args.output,
        {"status": "ERROR", "error": repr(error), "traceback": traceback.format_exc()},
    )
finally:
    simulation_app.close()

raise SystemExit(exit_code)
