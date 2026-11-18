#!/usr/bin/env python3
"""What units is the site USD in, and how does a local coordinate map to world? Read-only.

Reports stage metersPerUnit, the transform chain under the site root, and converts any
supplied local point with the real local-to-world matrix rather than a guessed scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Uav-Building-Inspection-v0")
parser.add_argument("--site_root", default="/World/ConstructionSite")
parser.add_argument("--local_point", type=float, nargs=3, action="append", default=None,
                    help="Point in the site USD's own coordinates. Repeatable.")
parser.add_argument("--prim", action="append", default=None,
                    help="Report the world transform of this prim path. Repeatable.")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import uav_indoor.tasks  # noqa: F401,E402

from pxr import Gf, Usd, UsdGeom  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    import isaacsim.core.utils.stage as stage_utils

    stage = stage_utils.get_current_stage()
    out = {
        "stage_meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "root_layer": stage.GetRootLayer().identifier,
        "sublayers": [],
        "chain": [],
    }
    for layer in stage.GetLayerStack():
        out["sublayers"].append({
            "identifier": layer.identifier,
            "metersPerUnit": layer.customLayerData.get("metersPerUnit")
            or (layer.pseudoRoot.GetCustomData().get("metersPerUnit") if layer.pseudoRoot else None),
        })

    # Transform chain from the stage root down to the site root.
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    prim = stage.GetPrimAtPath(args_cli.site_root)
    out["site_root_exists"] = bool(prim and prim.IsValid())
    node = prim
    while node and node.IsValid() and str(node.GetPath()) != "/":
        x = UsdGeom.Xformable(node)
        ops = []
        if x:
            for op in x.GetOrderedXformOps():
                ops.append({"op": op.GetOpName(), "value": str(op.Get())})
        out["chain"].append({"path": str(node.GetPath()), "xform_ops": ops})
        node = node.GetParent()
    out["chain"].reverse()

    l2w = xform_cache.GetLocalToWorldTransform(prim)
    out["site_local_to_world"] = [[round(l2w[r][c], 6) for c in range(4)] for r in range(4)]
    scale = [
        round(Gf.Vec3d(l2w[0][0], l2w[0][1], l2w[0][2]).GetLength(), 8),
        round(Gf.Vec3d(l2w[1][0], l2w[1][1], l2w[1][2]).GetLength(), 8),
        round(Gf.Vec3d(l2w[2][0], l2w[2][1], l2w[2][2]).GetLength(), 8),
    ]
    out["site_world_scale"] = scale

    out["converted_points"] = []
    for p in args_cli.local_point or []:
        w = l2w.Transform(Gf.Vec3d(*p))
        naive = [v * scale[0] for v in p]
        out["converted_points"].append({
            "local": list(p),
            "world_via_matrix": [round(w[i], 4) for i in range(3)],
            "world_via_scale_only": [round(v, 4) for v in naive],
        })
        print(f"local {p} -> world {[round(w[i],4) for i in range(3)]}   "
              f"(scale-only would give {[round(v,4) for v in naive]})", flush=True)

    out["prims"] = []
    for path in args_cli.prim or []:
        pr = stage.GetPrimAtPath(path)
        if not pr or not pr.IsValid():
            out["prims"].append({"path": path, "exists": False})
            print(f"prim {path}: NOT FOUND", flush=True)
            continue
        m = xform_cache.GetLocalToWorldTransform(pr)
        t = m.ExtractTranslation()
        out["prims"].append({"path": path, "exists": True,
                             "world_translation": [round(t[i], 4) for i in range(3)]})
        print(f"prim {path}: world {[round(t[i],4) for i in range(3)]}", flush=True)

    print(f"stage metersPerUnit = {out['stage_meters_per_unit']}  up = {out['stage_up_axis']}", flush=True)
    print(f"site world scale     = {scale}", flush=True)
    Path(args_cli.out).write_text(json.dumps(out, indent=2))
    print(f"PROBE_WROTE {args_cli.out}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    status = main()
    simulation_app.close()
    raise SystemExit(status)
