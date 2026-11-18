#!/usr/bin/env python3
"""Qualify the existing PX4 position path with fixed targets in open air.

This bypasses mission, policy, perception, shield, and termination updates. It calls
the production action apply method and writes its wrenches before each physics step.
Targets are fixed for each case. Results apply to this control path and configuration.
The full mission still requires separate qualification.
"""

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--seconds", type=float, default=12)
parser.add_argument("--settle_s", type=float, default=6)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--waypoint_boundary", action="store_true",
                    help="Exercise process_actions at 10 Hz with fixed mission references and hold residuals.")
parser.add_argument("--cases", nargs="+", default=None)
parser.add_argument("--result", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.repeats < 1 or args.seconds < args.settle_s + 2:
    parser.error("require environments, repeats, and at least two seconds after settling")
args.enable_cameras = True
app = AppLauncher(args)
simulation_app = app.app

import gymnasium as gym
import numpy as np
import torch
import uav_indoor.tasks  # noqa: F401
from fixed_target_metrics import LIMITS, qualify
from uav_indoor.tasks.manager_based.uav_indoor.inspection import MissionPhase

from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz

from isaaclab_tasks.utils import parse_env_cfg

CASES = (
    ("hold", (0, 0, 0), 0, False),
    ("heading_plus", (0, 0, 0), 20, False),
    ("heading_minus", (0, 0, 0), -20, False),
    ("forward_step", (1, 0, 0), 0, False),
    ("left_step", (0, 1, 0), 0, False),
    ("combined_step", (0.7, 0.7, 0.3), 20, False),
    ("disturbed_hold", (0, 0, 0), 0, True),
)

def main():
    path = Path(args.result)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg("Template-Uav-Building-Inspection-v0", device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.actions.px4_waypoint.randomize = False
    env = gym.make("Template-Uav-Building-Inspection-v0", cfg=cfg)
    task = env.unwrapped
    robot = task.scene["robot"]
    term = task.action_manager.get_term("px4_waypoint")
    dt = float(task.physics_dt)
    device, n = task.device, args.num_envs
    cases = [case for case in CASES if args.cases is None or case[0] in args.cases]
    if not cases or (args.cases and set(args.cases) - {case[0] for case in cases}):
        raise ValueError("unknown or empty case selection")
    out = {"verdict": "INCOMPLETE", "scope": "fixed-target production apply_actions, no mission",
           "waypoint_boundary": args.waypoint_boundary,
           "seed": args.seed, "physics_dt": dt, "limits": LIMITS, "cases": [],
           "episodes": 0, "requested_episodes": len(cases) * args.repeats * n,
           "motor_torque_sign": list(cfg.actions.px4_waypoint.motor_torque_sign),
           "motor_tau": cfg.actions.px4_waypoint.motor_tau,
           "k_torque": cfg.actions.px4_waypoint.k_torque,
           "k_thrust": cfg.actions.px4_waypoint.k_thrust,
           "visual_spin_joints": cfg.actions.px4_waypoint.visual_spin_joints}
    import inspect
    source = Path(inspect.getfile(type(term)))
    out["action_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    out["diagnostic_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out["grader_sha256"] = hashlib.sha256(Path(__file__).with_name("fixed_target_metrics.py").read_bytes()).hexdigest()
    path.write_text(json.dumps(out, indent=2))
    try:
        with torch.inference_mode():
            env.reset()
            for label, offset, yaw_deg, disturbed in cases:
                for repeat in range(args.repeats):
                    task.scene.reset()
                    pose = torch.zeros((n, 7), device=device)
                    pose[:, :3] = torch.tensor((-40., -30., 20.), device=device) + task.scene.env_origins
                    pose[:, 1] += torch.arange(n, device=device) * 2
                    initial_yaw = torch.linspace(-math.pi / 2, math.pi / 2, n, device=device)
                    roll = torch.full((n,), math.radians(10) if disturbed else 0., device=device)
                    zero = torch.zeros(n, device=device)
                    pose[:, 3:] = quat_from_euler_xyz(roll, zero, initial_yaw)
                    velocity = torch.zeros((n, 6), device=device)
                    if disturbed:
                        velocity[:, 0] = 0.3
                    robot.write_root_pose_to_sim(pose)
                    robot.write_root_velocity_to_sim(velocity)
                    joints = torch.zeros((n, robot.num_joints), device=device)
                    robot.write_joint_state_to_sim(joints, joints)
                    term.reset()
                    target = pose[:, :3] - task.scene.env_origins + torch.tensor(offset, device=device)
                    heading = initial_yaw + math.radians(yaw_deg)
                    term._target_w[:] = target
                    term._yaw_setpoint[:] = heading
                    phase = MissionPhase.TRANSIT if "step" in label else MissionPhase.OBSERVE
                    task._reference = replace(task._reference, position_w=target.clone(), yaw_w=heading.clone(),
                                              phase=torch.full((n,), int(phase), device=device, dtype=torch.long))
                    # A nonzero translation residual must not move an observation target.
                    residual = torch.zeros((n, 4), device=device)
                    if phase == MissionPhase.OBSERVE:
                        residual[:, :3] = torch.tensor((1., -1., 1.), device=device)
                    samples = {key: [] for key in ("position", "target", "angles", "heading", "rates", "throttle",
                                                   "waypoint_command", "heading_command", "safe_action")}
                    timestamps = []
                    start = task.sim.current_time
                    for step in range(round(args.seconds / dt)):
                        if args.waypoint_boundary and step % task.cfg.decimation == 0:
                            task._mission_step = task.common_step_counter
                            term.process_actions(residual)
                        term.apply_actions()
                        task.scene.write_data_to_sim()
                        task.sim.step(render=False)
                        task.scene.update(dt)
                        d = robot.data
                        angles = torch.stack(euler_xyz_from_quat(d.root_quat_w), dim=-1)
                        angles = torch.atan2(torch.sin(angles), torch.cos(angles))
                        for key, value in (
                            ("position", d.root_pos_w - task.scene.env_origins),
                            ("target", task._reference.position_w if args.waypoint_boundary else term._target_w),
                            ("heading", task._reference.yaw_w if args.waypoint_boundary else term._yaw_setpoint),
                            ("angles", angles), ("rates", d.root_ang_vel_b), ("throttle", term._throttle),
                            ("waypoint_command", term._target_w), ("heading_command", term._yaw_setpoint),
                            ("safe_action", term._safe_actions),
                        ):
                            samples[key].append(value.clone())
                        timestamps.append(task.sim.current_time - start)
                    data = {key: torch.stack(values).cpu().numpy() for key, values in samples.items()}
                    data["time"] = np.asarray(timestamps)
                    result = qualify(data, dt, args.settle_s)
                    result.update(case=label, repeat=repeat, samples=len(timestamps), seconds=args.seconds)
                    trace_path = path.with_name(f"{path.stem}_{label}_{repeat}.npz")
                    np.savez_compressed(trace_path, **data)
                    result["trace"] = str(trace_path)
                    out["cases"].append(result)
                    out["episodes"] += n
                    path.write_text(json.dumps(out, indent=2))
                    print(f"FIXED_TARGET {label} repeat={repeat} {result['verdict']} "
                          f"per_env={result['per_env_pass']} metrics={result.get('per_env_metrics')} "
                          f"invalid={result['invalid']}", flush=True)
        out["failures"] = [f"{r['case']} repeat {r['repeat']}: {r.get('per_env_failures', r['invalid'])}"
                           for r in out["cases"] if r["verdict"] != "PASS"]
        out["verdict"] = "FAIL" if out["failures"] else "PASS"
        path.write_text(json.dumps(out, indent=2))
        print(f"VERDICT {out['verdict']} {out['episodes']}/{out['requested_episodes']}", flush=True)
    finally:
        env.close()
    return 0 if out["verdict"] == "PASS" else 1

if __name__ == "__main__":
    status = main()
    simulation_app.close()
    raise SystemExit(status)
