#!/usr/bin/env python3
"""Can PX4 position control hold a fixed point? Read-only, no repo behaviour changed.

Swaps the action term's controller for `ParallelPosControl`, which the loaded binary
already exposes and which runs the full cascade (`CTRL_ALL`). Feeds it one target
captured once and never moved.

Both the measured position and the setpoint are environment-local. The shipped code
sends absolute world position to `set_status`, which the velocity path ignored because
it discards `pos_sp`. Position control would fly to the wrong place.

MPC_XY_P = 3.5 and MPC_Z_P = 5.0 have never executed in this project.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Uav-Building-Inspection-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--hold_steps", type=int, default=200)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--record_every", type=int, default=12,
                    help="Store position every Nth step. Statistics stay per-step.")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import uav_indoor.tasks  # noqa: F401,E402
from rlPx4Controller.pyParallelControl import ParallelPosControl  # noqa: E402

from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def never(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    for name in ("mission_complete", "collision", "recovery_failed",
                 "entry_success", "unauthorized_crossing"):
        t = getattr(env_cfg.terminations, name)
        t.func = never
        t.params = {}
    env = gym.make(args_cli.task, cfg=env_cfg)
    task = env.unwrapped
    robot = task.scene["robot"]
    term = task.action_manager.get_term("px4_waypoint")
    n, dev = args_cli.num_envs, task.device
    physics_hz = 1.0 / float(task.physics_dt)

    term._px4 = ParallelPosControl(n)
    target = torch.zeros((n, 3), device=dev)
    target_yaw = torch.zeros(n, device=dev)
    samples: list[list] = []
    sent: list[np.ndarray] = []

    def hold_apply() -> None:
        """The shipped motor path, but the setpoint is a fixed env-local position."""
        data = robot.data
        origin = task.scene.env_origins
        local = data.root_pos_w - origin
        height = local[:, 2]
        term._in_air |= height > term.cfg.in_air_height
        setpoint = torch.cat((target, target_yaw[:, None]), dim=-1)
        host = torch.cat(
            (local, data.root_quat_w, data.root_lin_vel_w, data.root_ang_vel_w,
             term._in_air.float()[:, None], setpoint), dim=-1
        ).detach().cpu().numpy()
        sent.append(host[:, 14:18].copy())
        term._px4.set_status(
            np.ascontiguousarray(host[:, 0:3]), np.ascontiguousarray(host[:, 3:7]),
            np.ascontiguousarray(host[:, 7:10]), np.ascontiguousarray(host[:, 10:13]),
            task.physics_dt, np.ascontiguousarray(host[:, 13]),
        )
        cmd = torch.as_tensor(term._px4.update(np.ascontiguousarray(host[:, 14:18])),
                              device=dev, dtype=torch.float32)
        term._throttle[:] = cmd.clamp(0.0, 1.0) * term.cfg.motor_scale
        omega = term._throttle_to_omega(term._throttle)
        thrust_cmd = abs(term.cfg.k_thrust) * term._thrust_scale * omega.square()
        init = ~term._thrust_initialized
        term._thrust_state[init] = thrust_cmd[init]
        term._thrust_initialized[init] = True
        alpha = task.physics_dt / (term._motor_tau + task.physics_dt)
        term._thrust_state += alpha * (thrust_cmd - term._thrust_state)
        torque = (abs(term.cfg.k_torque) / abs(term.cfg.k_thrust)) * term._thrust_state
        term._forces.zero_()
        term._forces[:, :-1, 2] = term._thrust_state
        term._torques.zero_()
        term._torques[:, -1, 2] = (term._motor_torque_sign * torque).sum(dim=-1)
        robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=term._forces, torques=term._torques,
            body_ids=term._wrench_body_ids, is_global=False,
        )
        robot.set_joint_effort_target(term._zero_joint_effort, joint_ids=term._joint_ids)

    stats: dict = {}

    def reset_stats() -> None:
        stats["roll_peak"] = torch.zeros(n, device=dev)
        stats["rate_sq"] = torch.zeros((n, 3), device=dev)
        stats["last_yaw"] = torch.zeros(n, device=dev)
        stats["steps"] = 0

    def tick(step: int) -> None:
        hold_apply()
        robot.write_data_to_sim()
        task.sim.step(render=False)
        task.scene.update(dt=task.physics_dt)
        d = robot.data
        roll, _, yaw = euler_xyz_from_quat(d.root_quat_w)
        # Running statistics and sparse position samples.
        stats["roll_peak"] = torch.maximum(stats["roll_peak"], wrap(roll).abs())
        stats["rate_sq"] += d.root_ang_vel_b.pow(2)
        stats["last_yaw"] = wrap(yaw)
        stats["steps"] += 1
        if step % args_cli.record_every == 0:
            samples.append((d.root_pos_w - task.scene.env_origins).clone())

    out = {"seed": args_cli.seed, "num_envs": n, "physics_hz": physics_hz,
           "controller": "ParallelPosControl (CTRL_ALL)", "frame": "env-local", "cases": []}
    try:
        with torch.inference_mode():
            for repeat in range(args_cli.repeats):
                env.reset()
                task.sim.step(render=False)
                task.scene.update(dt=task.physics_dt)
                target[:] = robot.data.root_pos_w - task.scene.env_origins
                _, _, yaw0 = euler_xyz_from_quat(robot.data.root_quat_w)
                target_yaw[:] = wrap(yaw0)
                samples.clear()
                sent.clear()
                reset_stats()
                for step in range(args_cli.hold_steps):
                    tick(step)
                if not samples:
                    raise AssertionError("no samples captured: the loop never ran")

                pos = torch.stack(samples)
                hz = stats["steps"] / (args_cli.hold_steps * float(task.physics_dt))
                err = torch.linalg.vector_norm(pos - target[None], dim=-1)
                stack = np.stack(sent)
                invalid = []
                if abs(hz - physics_hz) > 0.1 * physics_hz:
                    invalid.append(f"sampled at {hz:.1f} Hz, not {physics_hz:.0f} Hz")
                moved = float(np.abs(stack - stack[0]).max())
                if moved > 1e-6:
                    invalid.append(f"the target moved by {moved:.6f} during the hold")
                half = len(samples) // 2
                first = err[:half].max(dim=0).values
                second = err[half:].max(dim=0).values
                rec = {
                    "repeat": repeat, "sample_hz": round(hz, 1), "invalid": invalid,
                    "seconds": args_cli.hold_steps * float(task.physics_dt),
                    "target_max_change": moved,
                    "per_env_err_final_m": err[-1].tolist(),
                    "per_env_err_max_m": err.max(dim=0).values.tolist(),
                    "per_env_growing": (second > 1.5 * first.clamp_min(1e-3)).tolist(),
                    "per_env_heading_drift_deg": (
                        wrap(stats["last_yaw"] - target_yaw).abs() * 180 / math.pi
                    ).tolist(),
                    "per_env_roll_peak_deg": (stats["roll_peak"] * 180 / math.pi).tolist(),
                    "per_env_body_rate_rms": (
                        stats["rate_sq"] / max(stats["steps"], 1)
                    ).sqrt().tolist(),
                    "control_hz": physics_hz,
                    "position_record_hz": physics_hz / args_cli.record_every,
                }
                out["cases"].append(rec)
                flag = "INVALID " + "; ".join(invalid) if invalid else (
                    "GROWING" if any(rec["per_env_growing"]) else "bounded")
                print(f"hold r{repeat} {hz:.0f}Hz {rec['seconds']:.0f}s  "
                      f"err final {[round(v,3) for v in rec['per_env_err_final_m']]} m  "
                      f"max {[round(v,3) for v in rec['per_env_err_max_m']]} m  "
                      f"heading {[round(v,1) for v in rec['per_env_heading_drift_deg']]} deg  {flag}",
                      flush=True)
    finally:
        env.close()

    out["failures"] = [f"r{c['repeat']} INVALID: {'; '.join(c['invalid'])}" for c in out["cases"] if c["invalid"]]
    out["verdict"] = "FAIL" if out["failures"] else "PASS"
    for f in out["failures"]:
        print(f"FAIL: {f}", flush=True)
    print(f"VERDICT {out['verdict']}", flush=True)
    Path(args_cli.out).write_text(json.dumps(out, indent=2))
    print(f"PROBE_WROTE {args_cli.out}", flush=True)
    return 0


if __name__ == "__main__":
    status = main()
    simulation_app.close()
    raise SystemExit(status)
