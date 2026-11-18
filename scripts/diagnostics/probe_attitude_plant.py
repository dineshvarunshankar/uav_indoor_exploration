#!/usr/bin/env python3
"""Attitude plant response at physics rate, per environment, with the shield off.

Every criterion applies per environment. A mean across environments can stay positive
while one of them turns the wrong way.

Each case validates itself before it reports. A case is INVALID, never a result, when
the sample rate is not the physics rate, or when the axis under test received no
command. An earlier version pinned OBSERVE for the translation cases; OBSERVE is motion
mode 3, whose XY and Z limits are 0.0, so those cases commanded nothing and would have
reported zeros as a measurement.

This measures the plant. It does not qualify closed-loop stability.
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
parser.add_argument("--env_steps", type=int, default=40)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--settle_frac", type=float, default=0.5)
parser.add_argument("--yaw_tol_rad_s", type=float, default=0.05)
parser.add_argument("--motor_torque_sign", type=float, nargs=4, default=None)
parser.add_argument("--trace_out", type=str, default=None)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import uav_indoor.tasks  # noqa: F401,E402
from uav_indoor.tasks.manager_based.uav_indoor.inspection import MissionPhase  # noqa: E402

from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# OBSERVE yaw and TRANSIT translation regimes.
CASES = [
    ("yaw_plus", "OBSERVE", (0.0, 0.0, 0.0, 1.0), 3, +1),
    ("yaw_minus", "OBSERVE", (0.0, 0.0, 0.0, -1.0), 3, -1),
    ("no_command", "OBSERVE", (0.0, 0.0, 0.0, 0.0), None, 0),
    ("forward_plus", "TRANSIT", (0.5, 0.0, 0.0, 0.0), 0, None),
    ("backward", "TRANSIT", (-0.5, 0.0, 0.0, 0.0), 0, None),
    ("left_plus", "TRANSIT", (0.0, 0.5, 0.0, 0.0), 1, None),
    ("right", "TRANSIT", (0.0, -0.5, 0.0, 0.0), 1, None),
    ("up", "TRANSIT", (0.0, 0.0, 0.5, 0.0), 2, None),
    ("down", "TRANSIT", (0.0, 0.0, -0.5, 0.0), 2, None),
]


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if args_cli.motor_torque_sign is not None:
        env_cfg.actions.px4_waypoint.motor_torque_sign = tuple(args_cli.motor_torque_sign)
    env = gym.make(args_cli.task, cfg=env_cfg)
    task = env.unwrapped
    robot = task.scene["robot"]
    term = task.action_manager.get_term("px4_waypoint")
    n = args_cli.num_envs
    physics_hz = 1.0 / float(task.physics_dt)

    out = {
        "seed": args_cli.seed,
        "num_envs": n,
        "motor_torque_sign": list(env_cfg.actions.px4_waypoint.motor_torque_sign),
        "physics_hz": physics_hz,
        "step_dt": float(task.step_dt),
        "cases": [],
    }
    traces: dict = {}
    samples: list[list] = []
    original = robot.write_data_to_sim

    def hooked() -> None:
        original()
        d = robot.data
        roll, pitch, yaw = euler_xyz_from_quat(d.root_quat_w)
        samples.append([
            (d.root_pos_w - task.scene.env_origins).clone(),
            torch.stack((wrap(roll), wrap(pitch), wrap(yaw)), dim=-1),
            d.root_ang_vel_b.clone(),
            term._yaw_setpoint.clone(),
            term._safe_actions.clone(),
            term.target_w.clone(),
        ])

    robot.write_data_to_sim = hooked
    try:
        with torch.inference_mode():
            for label, phase_name, residual, axis, expect in CASES:
                phase = int(getattr(MissionPhase, phase_name))
                for repeat in range(args_cli.repeats):
                    env.reset()
                    samples.clear()
                    target_errors = []
                    action = torch.tensor(residual, device=task.device).repeat(n, 1)
                    sat0 = int(term._action_saturated_total.sum().item())
                    for _ in range(args_cli.env_steps):
                        task._supervisor.phase[:] = phase
                        task._supervisor.phase_time[:] = 0.0
                        if phase_name == "TRANSIT":
                            here = robot.data.root_pos_w - task.scene.env_origins
                            task.building_standoff_w[:] = here
                        before_position = robot.data.root_pos_w - task.scene.env_origins
                        _, _, before_yaw = euler_xyz_from_quat(robot.data.root_quat_w)
                        env.step(action)

                        safe_now = term.safe_actions
                        reach = torch.tensor(
                            (
                                term.cfg.v_max_xy * term.cfg.reach_seconds,
                                term.cfg.v_max_xy * term.cfg.reach_seconds,
                                term.cfg.v_max_z * term.cfg.reach_seconds,
                            ),
                            device=task.device,
                        )
                        offset_b = safe_now[:, :3] * reach
                        cosine = torch.cos(before_yaw)
                        sine = torch.sin(before_yaw)
                        offset_w = torch.stack(
                            (
                                cosine * offset_b[:, 0] - sine * offset_b[:, 1],
                                sine * offset_b[:, 0] + cosine * offset_b[:, 1],
                                offset_b[:, 2],
                            ),
                            dim=-1,
                        )
                        expected = before_position + offset_w
                        if phase_name == "OBSERVE":
                            expected = task.mission_reference.position_w
                        target_errors.append(
                            torch.linalg.vector_norm(term.target_w - expected, dim=-1)
                        )
                    pos = torch.stack([s[0] for s in samples])
                    att = torch.stack([s[1] for s in samples])
                    rate = torch.stack([s[2] for s in samples])
                    yawsp = torch.stack([s[3] for s in samples])
                    safe = torch.stack([s[4] for s in samples])
                    target = torch.stack([s[5] for s in samples])
                    target_error = torch.stack(target_errors)
                    cut = int(len(samples) * args_cli.settle_frac)
                    control_cut = int(len(target_errors) * args_cli.settle_frac)
                    settled = rate[cut:]
                    per_env = settled[:, :, 2].mean(dim=0)
                    hz = len(samples) / (args_cli.env_steps * float(task.step_dt))
                    cmd = safe[cut:].abs().mean(dim=0)

                    invalid = []
                    if abs(hz - physics_hz) > 0.1 * physics_hz:
                        invalid.append(f"sampled at {hz:.1f} Hz, not the {physics_hz:.0f} Hz physics rate")
                    if axis is not None and float(cmd[:, axis].min()) < 0.05:
                        invalid.append(
                            f"axis {axis} received no command: min |safe_action| "
                            f"{float(cmd[:, axis].min()):.4f}"
                        )
                    if axis is None and float(cmd.abs().max()) > 0.01:
                        invalid.append(f"a no-command case still commanded {float(cmd.abs().max()):.4f}")

                    record = {
                        "case": label,
                        "phase": phase_name,
                        "repeat": repeat,
                        "residual": list(residual),
                        "axis_under_test": axis,
                        "samples": len(samples),
                        "sample_hz": round(hz, 1),
                        "invalid": invalid,
                        "mean_abs_safe_action": cmd.mean(dim=0).tolist(),
                        "target_travel_m": float(
                            torch.linalg.vector_norm(target[-1] - target[cut], dim=-1).max()
                        ),
                        "per_env_target_contract_max_error_m": (
                            target_error[control_cut:].max(dim=0).values.tolist()
                        ),
                        "per_env_settled_yaw_rate": per_env.tolist(),
                        "per_env_drift_max_m": torch.linalg.vector_norm(
                            pos - pos[0], dim=-1
                        ).max(dim=0).values.tolist(),
                        "per_env_roll_peak_deg": (
                            att[cut:, :, 0].abs().max(dim=0).values * 180 / math.pi
                        ).tolist(),
                        "per_env_pitch_peak_deg": (
                            att[cut:, :, 1].abs().max(dim=0).values * 180 / math.pi
                        ).tolist(),
                        "per_env_body_rate_rms": settled.pow(2).mean(dim=0).sqrt().tolist(),
                        "per_env_yaw_setpoint_error_rad": wrap(
                            yawsp[cut:] - att[cut:, :, 2]
                        ).abs().mean(dim=0).tolist(),
                        "saturated_steps": int(term._action_saturated_total.sum().item()) - sat0,
                    }
                    if invalid:
                        record["per_env_pass"] = None
                    elif expect is None:
                        record["per_env_pass"] = [
                            value <= 1.0e-4
                            for value in record["per_env_target_contract_max_error_m"]
                        ]
                    elif expect == 0:
                        record["per_env_pass"] = [abs(v) < args_cli.yaw_tol_rad_s for v in per_env.tolist()]
                    else:
                        record["per_env_pass"] = [
                            v * expect > args_cli.yaw_tol_rad_s for v in per_env.tolist()
                        ]
                    record["all_envs_pass"] = (
                        None if record["per_env_pass"] is None else all(record["per_env_pass"])
                    )
                    out["cases"].append(record)
                    traces[f"{label}_{repeat}"] = {
                        "yaw_rate": rate[:, :, 2].tolist(),
                        "yaw": att[:, :, 2].tolist(),
                        "yaw_setpoint": yawsp.tolist(),
                    }
                    flag = "INVALID " + "; ".join(invalid) if invalid else f"pass {record['per_env_pass']}"
                    print(
                        f"{label:13s} {phase_name:8s} r{repeat} {hz:6.1f} Hz  "
                        f"yaw_rate/env {[round(v, 3) for v in per_env.tolist()]}  {flag}",
                        flush=True,
                    )
    finally:
        robot.write_data_to_sim = original
        env.close()

    invalid_cases = [c for c in out["cases"] if c["invalid"]]
    graded = [c for c in out["cases"] if c["all_envs_pass"] is not None]
    out["invalid_cases"] = len(invalid_cases)
    out["failures"] = [
        f"{c['case']} repeat {c['repeat']}: per-env yaw rate {c['per_env_settled_yaw_rate']}"
        for c in graded if not c["all_envs_pass"]
    ] + [f"{c['case']} repeat {c['repeat']} INVALID: {'; '.join(c['invalid'])}" for c in invalid_cases]
    out["verdict"] = "FAIL" if out["failures"] or not graded else "PASS"
    for f in out["failures"]:
        print(f"FAIL: {f}", flush=True)
    print(f"VERDICT {out['verdict']}  graded {len(graded)}  invalid {len(invalid_cases)}", flush=True)
    Path(args_cli.out).write_text(json.dumps(out, indent=2))
    if args_cli.trace_out:
        Path(args_cli.trace_out).write_text(json.dumps(traces))
    print(f"PROBE_WROTE {args_cli.out}", flush=True)
    return 0


if __name__ == "__main__":
    status = main()
    simulation_app.close()
    raise SystemExit(status)
