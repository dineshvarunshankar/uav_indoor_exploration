"""Check clear and overlapping contact for each procedural obstacle family."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher({"headless": True, "enable_cameras": True}).app

import argparse
import json
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("/tmp/obstacle_navigation_colliders.json"))
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
exit_code = 0

try:
    import gymnasium as gym
    import torch
    import uav_indoor.tasks  # noqa: F401

    from isaaclab_tasks.utils import parse_env_cfg

    def never(env):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    cfg = parse_env_cfg("Template-Uav-Obstacle-Navigation-v0", device="cuda:0", num_envs=4)
    cfg.seed = 0
    cfg.initial_difficulty = 3
    cfg.layout_bank_size = 4
    cfg.terminations.collision.func = never
    cfg.terminations.goal_reached.func = never
    cfg.terminations.time_out.func = never
    env = gym.make("Template-Uav-Obstacle-Navigation-v0", cfg=cfg)
    task = env.unwrapped
    env.reset()
    ids = torch.arange(4, device=task.device)
    # Dense layouts for all collider slots.
    dense_bank = task.training_layout_bank[3]
    for difficulty in task.training_layout_bank:
        task.training_layout_bank[difficulty] = dense_bank
    task._reset_idx(ids)
    selected = task.obstacle_active[ids, ids]
    if not bool(selected.all()):
        raise RuntimeError(f"selected collider slots are inactive: {selected.tolist()}")
    zero = torch.zeros(4, 4, device=task.device)

    def hold(position_w):
        pose = torch.zeros((4, 7), device=task.device)
        pose[:, :3] = position_w
        pose[:, 3] = 1.0
        task.scene["robot"].write_root_pose_to_sim(pose, env_ids=ids)
        task.scene["robot"].write_root_velocity_to_sim(torch.zeros((4, 6), device=task.device), env_ids=ids)
        task.sim.step(render=False)
        task.scene.update(dt=task.physics_dt)

    clear_position = task.scene.env_origins + torch.tensor([0.0, 0.0, 9.0], device=task.device)
    for _ in range(12):
        hold(clear_position)
    clear_force = torch.linalg.vector_norm(task.scene["contact_forces"].data.net_forces_w_history, dim=-1).amax(
        dim=(1, 2)
    )

    overlap_position = task.scene.env_origins.clone()
    for index in range(4):
        overlap_position[index] += task.obstacle_centres_w[index, index]
    peak_force = torch.zeros(4, device=task.device)
    for _ in range(8):
        hold(overlap_position)
        force = torch.linalg.vector_norm(task.scene["contact_forces"].data.net_forces_w_history, dim=-1).amax(
            dim=(1, 2)
        )
        peak_force = torch.maximum(peak_force, force)

    object_positions = task.scene["obstacles"].data.object_pos_w[ids, ids]
    expected_positions = task.scene.env_origins + task.obstacle_centres_w[ids, ids]
    placement_error = torch.linalg.vector_norm(object_positions - expected_positions, dim=-1)
    families = ("box", "wall", "post", "overhead_slab")
    records = [
        {
            "environment": index,
            "family": families[index],
            "clear_contact_force_n": float(clear_force[index]),
            "overlap_contact_force_n": float(peak_force[index]),
            "placement_error_m": float(placement_error[index]),
            "passed": bool(clear_force[index] <= 1.0 and peak_force[index] > 1.0 and placement_error[index] <= 1.0e-5),
        }
        for index in range(4)
    ]
    args.output.write_text(
        json.dumps({"status": "PASS" if all(r["passed"] for r in records) else "FAIL", "records": records}, indent=2)
    )
    env.close()
except Exception as error:
    exit_code = 1
    args.output.write_text(
        json.dumps({"status": "ERROR", "error": repr(error), "traceback": traceback.format_exc()}, indent=2)
    )
finally:
    simulation_app.close()

raise SystemExit(exit_code)
