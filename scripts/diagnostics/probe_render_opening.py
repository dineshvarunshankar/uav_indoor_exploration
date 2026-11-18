#!/usr/bin/env python3
"""Look at a point on a facade through the aircraft camera. Read-only.

Places the aircraft outside a facade at several standoffs, points it at the wall, and
saves the RGB frame. Use it to check a measured opening position by eye.

Coordinates are metres in the Isaac stage. The site USD authors in centimetres, so
divide a value read from the USD by 100.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Uav-Building-Inspection-v0")
parser.add_argument("--point", type=float, nargs=3, action="append", required=True,
                    help="World point in metres. Repeatable.")
parser.add_argument("--facade", action="append", required=True,
                    choices=["south", "west", "north", "east"],
                    help="Which facade the point sits on. One per --point.")
parser.add_argument("--label", action="append", default=None)
parser.add_argument("--standoffs", type=float, nargs="+", default=[2.0, 4.0, 8.0])
parser.add_argument("--settle", type=int, default=40)
parser.add_argument("--out_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import uav_indoor.tasks  # noqa: F401,E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# outward normal, and the yaw that points the nose at the wall
FACADE = {
    "south": ((0.0, -1.0), math.pi / 2),
    "north": ((0.0, 1.0), -math.pi / 2),
    "west": ((-1.0, 0.0), 0.0),
    "east": ((1.0, 0.0), math.pi),
}


def never(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    for name in ("mission_complete", "collision", "recovery_failed",
                 "entry_success", "unauthorized_crossing"):
        t = getattr(env_cfg.terminations, name)
        t.func = never
        t.params = {}
    env = gym.make(args_cli.task, cfg=env_cfg)
    task = env.unwrapped
    robot = task.scene["robot"]
    cam = task.scene["rgb_camera"]
    origin = task.scene.env_origins[0]
    dev = task.device
    env.reset()

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = args_cli.label or [f"point{i}" for i in range(len(args_cli.point))]
    manifest = []

    from PIL import Image

    for point, facade, label in zip(args_cli.point, args_cli.facade, labels):
        (nx, ny), yaw = FACADE[facade]
        for standoff in args_cli.standoffs:
            x = point[0] + nx * standoff
            y = point[1] + ny * standoff
            z = point[2]
            pose = torch.zeros((1, 7), device=dev)
            pose[0, :3] = torch.tensor([x, y, z], device=dev) + origin
            pose[0, 3] = math.cos(yaw / 2.0)
            pose[0, 6] = math.sin(yaw / 2.0)
            for _ in range(args_cli.settle):
                robot.write_root_pose_to_sim(pose)
                robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
                task.sim.step(render=True)
                task.scene.update(dt=task.physics_dt)
            rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)
            rgb = rgb[..., :3]
            name = f"{label}_{facade}_{standoff:g}m.png"
            Image.fromarray(rgb).save(out_dir / name)
            manifest.append({
                "label": label, "facade": facade, "target_m": list(point),
                "camera_pos_m": [round(x, 3), round(y, 3), round(z, 3)],
                "yaw_rad": round(yaw, 4), "standoff_m": standoff,
                "file": str(out_dir / name), "image_shape": list(rgb.shape),
                "clearance_m": round(float(task.clearance_m[0]), 3),
                "clearance_known": bool(task.clearance_known[0]),
            })
            print(f"{label:16s} {facade:5s} standoff {standoff:4.1f} m  cam "
                  f"({x:7.2f},{y:7.2f},{z:6.2f})  clearance "
                  f"{float(task.clearance_m[0]):6.2f} known={bool(task.clearance_known[0])}  -> {name}",
                  flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"PROBE_WROTE {out_dir}/manifest.json", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    status = main()
    simulation_app.close()
    raise SystemExit(status)
