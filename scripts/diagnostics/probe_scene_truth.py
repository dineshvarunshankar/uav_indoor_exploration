#!/usr/bin/env python3
"""Dump the simulator facts the repository does not contain. Read-only.

The task's assets live on a remote Nucleus server with no local copy, so mass, inertia,
camera extrinsics and building geometry can only be read from a running stage.

    python scripts/diagnostics/probe_scene_truth.py --headless --out <path>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Uav-Building-Inspection-v0")
parser.add_argument("--site_colliders", type=int, default=0)
parser.add_argument("--site_root", default="/World/ConstructionSite")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import uav_indoor.tasks  # noqa: F401,E402
from uav_indoor.tasks.manager_based.uav_indoor.building_inspection.site_colliders import (  # noqa: E402
    LAST_REPORT,
)
from uav_indoor.tasks.manager_based.uav_indoor.building_inspection.site_colliders import (  # noqa: E402
    enable_on as enable_site_colliders,
)

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.site_colliders.enabled = bool(args_cli.site_colliders)
    if env_cfg.site_colliders.enabled:
        enable_site_colliders(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    task = env.unwrapped
    env.reset()

    import isaacsim.core.utils.stage as stage_utils

    stage = stage_utils.get_current_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    def bbox(prim):
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if r.IsEmpty():
            return None
        lo, hi = r.GetMin(), r.GetMax()
        return [round(float(v), 3) for v in (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])]

    out: dict = {"task": args_cli.task, "site_colliders_enabled": bool(args_cli.site_colliders)}

    out["colliders"] = [
        {"path": str(p.GetPath()), "bbox": bbox(p)}
        for p in stage.Traverse()
        if p.HasAPI(UsdPhysics.CollisionAPI)
    ][:400]
    out["collider_count"] = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))
    out["rigid_body_count"] = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))

    root = stage.GetPrimAtPath(args_cli.site_root)
    proxies = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    out["site"] = {
        "root": args_cli.site_root,
        "meshes_direct": sum(1 for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)),
        "meshes_with_instance_proxies": sum(1 for p in Usd.PrimRange(root, proxies) if p.IsA(UsdGeom.Mesh)),
        "instanced_prims": sum(1 for p in Usd.PrimRange(root) if p.IsInstance()),
    }
    out["buildings"] = {}
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if not name.lower().startswith("building"):
            continue
        meshes = [
            {"name": m.GetName(), "path": str(m.GetPath()), "bbox": bbox(m),
             "triangles": sum(max(int(c) - 2, 0) for c in (UsdGeom.Mesh(m).GetFaceVertexCountsAttr().Get() or []))}
            for m in Usd.PrimRange(prim) if m.IsA(UsdGeom.Mesh)
        ]
        out["buildings"][name] = {
            "bbox": bbox(prim),
            "mesh_count": len(meshes),
            "triangles": sum(m["triangles"] for m in meshes),
            "meshes": meshes,
        }

    robot = task.scene["robot"]
    view = robot.root_physx_view
    out["vehicle"] = {
        "body_names": list(robot.body_names),
        "masses": view.get_masses()[0].tolist(),
        "total_mass_kg": float(view.get_masses()[0].sum()),
        "inertias": view.get_inertias()[0].tolist(),
        "coms": view.get_coms()[0].tolist(),
    }
    for name in ("tof_camera", "rgb_camera"):
        cam = task.scene[name]
        out[name] = {
            "pos_w": cam.data.pos_w[0].tolist(),
            "pos_rel_to_root": (cam.data.pos_w[0] - robot.data.root_pos_w[0]).tolist(),
            "quat_w_ros_wxyz": cam.data.quat_w_ros[0].tolist(),
            "intrinsic_matrix": cam.data.intrinsic_matrices[0].tolist(),
            "image_shape": list(cam.image_shape),
            "update_period": float(cam.cfg.update_period),
        }
    out["timing"] = {
        "sim_dt": float(env_cfg.sim.dt),
        "decimation": int(env_cfg.decimation),
        "step_dt": float(task.step_dt),
        "render_interval": int(env_cfg.sim.render_interval),
    }
    import importlib
    out["controller_modules"] = {}
    for mod in ("rlPx4Controller.pyParallelControl", "rlPx4Controller.pyControl"):
        try:
            out["controller_modules"][mod] = importlib.import_module(mod).__file__
        except Exception as exc:  # noqa: BLE001
            out["controller_modules"][mod] = f"IMPORT FAILED: {exc!r}"
    report = LAST_REPORT.get(args_cli.site_root)
    out["collider_report"] = report.summary() if report else None

    Path(args_cli.out).write_text(json.dumps(out, indent=2))
    print(f"PROBE_WROTE {args_cli.out}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    status = main()
    simulation_app.close()
    raise SystemExit(status)
