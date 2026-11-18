#!/usr/bin/env python3
"""Does the collider-enabled site stop the aircraft, and still let it through an opening?

Building02 is an unfinished concrete frame: its openings are large floor-to-slab gaps,
not 1.8 m windows, and it has only two floor slabs. So this gate scans for solid
material and takes its test points from the scan.

`--site_colliders 0` confirms the original scene is unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Uav-Building-Inspection-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--site_colliders", type=int, default=1)
parser.add_argument("--approximation", default="none")
parser.add_argument("--speeds", type=float, nargs="+", default=[1.0, 3.0, 6.0])
parser.add_argument("--travel_m", type=float, default=2.5)
parser.add_argument("--scan_reach_m", type=float, default=4.0)
parser.add_argument("--scan_step_m", type=float, default=0.25)
parser.add_argument("--settle_steps", type=int, default=8)
parser.add_argument("--result", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import uav_indoor.tasks  # noqa: F401,E402
from uav_indoor.tasks.manager_based.uav_indoor.building_inspection.mdp.terminations import (  # noqa: E402
    collision as collision_event,
)
from uav_indoor.tasks.manager_based.uav_indoor.building_inspection.site_colliders import (  # noqa: E402
    LAST_REPORT,
    open_run_through_centre,
    strongest_contact_offset,
)
from uav_indoor.tasks.manager_based.uav_indoor.building_inspection.site_colliders import (  # noqa: E402
    enable_on as enable_site_colliders,
)

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

WALL_X = 0.49
PARK = (-40.0, -30.0, 12.0)
# Truth rows from west_face_openings.yaml, three openings on three floors.
OPENINGS = [(10.221, 3.031), (5.707, 7.099), (13.623, 10.860)]


class Rig:
    """Place the aircraft at a pose and report whether the site pushed back."""

    def __init__(self, task) -> None:
        self.task = task
        self.robot = task.scene["robot"]
        self.sensor = task.scene.sensors["contact_forces"]
        self.origin = task.scene.env_origins
        self.dt = task.physics_dt
        self.n = int(task.num_envs)

    def _write(self, x: float, y: float, z: float) -> None:
        pose = torch.zeros((self.n, 7), device=self.task.device)
        pose[:, :3] = torch.tensor([x, y, z], device=self.task.device) + self.origin
        pose[:, 3] = 1.0
        self.robot.write_root_pose_to_sim(pose)
        self.robot.write_root_velocity_to_sim(torch.zeros((self.n, 6), device=self.task.device))

    def _tick(self) -> None:
        self.task.sim.step(render=False)
        self.task.scene.update(dt=self.dt)

    def force(self) -> float:
        return float(torch.linalg.vector_norm(self.sensor.data.net_forces_w_history, dim=-1).amax())

    def flush(self) -> None:
        """Park in open air until the contact history clears."""

        for _ in range(12):
            self._write(*PARK)
            self._tick()
            if self.force() == 0.0 and not bool(collision_event(self.task).any()):
                return
        raise AssertionError("contact history did not clear in open air")

    def probe(self, x: float, y: float, z: float) -> tuple[bool, float, bool]:
        """Hold one pose and report (collided, peak force, every env agreed).

        One step after a teleport is too early for PhysX to report contact.
        """
        self.flush()
        self._write(x, y, z)
        peak = 0.0
        hits = 0
        for _ in range(args_cli.settle_steps):
            self._write(x, y, z)
            self._tick()
            peak = max(peak, self.force())
            hit = collision_event(self.task)
            hits = max(hits, int(hit.sum().item()))
        return hits > 0, peak, hits in (0, self.n)

    def sweep(self, y: float, z: float, speed: float) -> dict:
        """Advance along +x one physics step at a time and report the first contact."""
        self.flush()
        step_m = speed * self.dt
        steps = int(args_cli.travel_m / step_m)
        x = WALL_X - args_cli.travel_m
        peak = 0.0
        for _ in range(steps):
            self._write(x, y, z)
            self._tick()
            peak = max(peak, self.force())
            hit = collision_event(self.task)
            if bool(hit.any()):
                agree = int(hit.sum().item()) == self.n
                out = {"collision_event": True, "peak_force_N": round(peak, 3),
                       "first_contact_x": round(x, 3), "all_envs_agree": agree, "steps": steps}
                self.flush()
                return out
            x += step_m
        return {"collision_event": False, "peak_force_N": round(peak, 3),
                "first_contact_x": None, "all_envs_agree": True, "steps": steps}


def scan_axis(rig: Rig, y: float, z: float, axis: str) -> dict:
    """Walk one axis through an opening and record where the facade is solid."""
    reach, step = args_cli.scan_reach_m, args_cli.scan_step_m
    solid, free, solid_samples = [], [], []
    offsets = [round(-reach + i * step, 3) for i in range(int(2 * reach / step) + 1)]
    for k in offsets:
        collided, peak, _ = rig.probe(
            WALL_X, y + (k if axis == "y" else 0.0), z + (k if axis == "z" else 0.0)
        )
        (solid if collided else free).append(k)
        if collided:
            solid_samples.append((k, peak))
    centred = open_run_through_centre(free, step)
    return {
        "axis": axis,
        "reach_m": reach,
        "solid_found": bool(solid),
        "solid_offsets": solid,
        "solid_samples": solid_samples,
        "open_span_m": round(centred[-1] - centred[0] + step, 3) if centred else 0.0,
        "open_from": centred[0] if centred else None,
        "open_to": centred[-1] if centred else None,
    }


def strongest_solid(scan: dict) -> float | None:
    """The scanned solid offset with the largest contact force."""

    return strongest_contact_offset(scan["solid_samples"])


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.site_colliders.enabled = bool(args_cli.site_colliders)
    env_cfg.site_colliders.approximation = args_cli.approximation
    # parse_env_cfg has already run __post_init__; re-running it would clobber overrides.
    if env_cfg.site_colliders.enabled:
        enable_site_colliders(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    task = env.unwrapped
    env.reset()
    rig = Rig(task)
    enabled = bool(args_cli.site_colliders)
    report = LAST_REPORT.get("/World/ConstructionSite")

    failures: list[str] = []
    profiles: list[dict] = []
    for oy, oz in OPENINGS:
        profile = {"opening_y": oy, "opening_z": oz}
        for axis in ("y", "z"):
            profile[axis] = scan_axis(rig, oy, oz, axis)
        profiles.append(profile)
        print(
            f"opening y={oy:.3f} z={oz:.3f}  open {profile['y']['open_span_m']:.2f} m across, "
            f"{profile['z']['open_span_m']:.2f} m tall  "
            f"solid y={profile['y']['solid_found']} z={profile['z']['solid_found']}",
            flush=True,
        )
        if not enabled:
            continue
        if profile["y"]["open_span_m"] <= 0.0 or profile["z"]["open_span_m"] <= 0.0:
            failures.append(f"opening at y={oy:.3f} z={oz:.3f} is sealed: the aircraft cannot enter")

    # Test points come from the scan, never from the nominal opening size.
    cases: list[tuple[str, tuple[float, float], bool]] = []
    for profile in profiles:
        oy, oz = profile["opening_y"], profile["opening_z"]
        cases.append((f"through_opening_z{oz:.1f}", (oy, oz), False))
        for axis, label in (("z", "slab_band"), ("y", "column_or_edge")):
            offset = strongest_solid(profile[axis])
            if offset is None:
                continue
            dy = offset if axis == "y" else 0.0
            dz = offset if axis == "z" else 0.0
            cases.append((f"{label}_z{oz:.1f}", (oy + dy, oz + dz), True))

    results = []
    for name, (y, z), expect_hit in cases:
        for speed in args_cli.speeds:
            outcome = rig.sweep(y, z, speed) | {
                "case": name, "speed_m_s": speed, "y": round(y, 3), "z": round(z, 3),
                "expected_collision": expect_hit,
            }
            results.append(outcome)
            print(
                f"{name:26s} {speed:4.1f} m/s  collision={outcome['collision_event']!s:5s} "
                f"peak={outcome['peak_force_N']:9.3f} N  expected={expect_hit}",
                flush=True,
            )
            if not enabled:
                continue
            if expect_hit and not outcome["collision_event"]:
                failures.append(f"{name} at {speed} m/s passed through solid material")
            if not expect_hit and outcome["collision_event"]:
                failures.append(f"{name} at {speed} m/s was blocked, but the opening must stay passable")
            if not outcome["all_envs_agree"]:
                failures.append(f"{name} at {speed} m/s: parallel environments disagreed")

    if not enabled:
        if any(r["collision_event"] for r in results):
            failures.append("the original scene reported a collision, so it is not unchanged")
    elif report is None:
        failures.append("the collider spawner did not run, so no site collider report exists")
    elif not report.applied:
        failures.append("the collider spawner applied no colliders")
    elif report.skipped_not_included:
        failures.append(f"{report.skipped_not_included} scene meshes were not selected")
    elif report.skipped_instance_proxy:
        failures.append(
            f"{len(report.skipped_instance_proxy)} selected meshes remain instance proxies"
        )
    elif not any(c[2] for c in cases):
        failures.append(
            f"the scan found no solid material within {args_cli.scan_reach_m:g} m of any opening, "
            "so no case tested whether a collider stops the aircraft"
        )

    for failure in failures:
        print(f"FAIL: {failure}", flush=True)
    verdict = "FAIL" if failures else "PASS"
    if not failures:
        print("PASS: solid material stops the aircraft, openings stay passable, every environment agrees",
              flush=True)

    if args_cli.result:
        Path(args_cli.result).write_text(json.dumps({
            "gate": "site_colliders",
            "verdict": verdict,
            "failures": failures,
            "requested_episodes": len(cases) * len(args_cli.speeds),
            "episodes": len(results),
            "site_colliders_enabled": enabled,
            "approximation": args_cli.approximation,
            "num_envs": rig.n,
            "collider_report": report.summary() if report else None,
            "scan_reach_m": args_cli.scan_reach_m,
            "scan_step_m": args_cli.scan_step_m,
            "openings_with_no_solid_material_in_reach": [
                {"y": p["opening_y"], "z": p["opening_z"]}
                for p in profiles
                if not (p["y"]["solid_found"] or p["z"]["solid_found"])
            ],
            "opening_profiles": profiles,
            "cases": results,
        }, indent=2))
    env.close()
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    status = main()
    simulation_app.close()
    raise SystemExit(status)
