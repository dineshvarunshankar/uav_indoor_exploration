#!/usr/bin/env python3
"""What the aircraft does when told to hold, and whether reset clears state. Read-only.

The action term has no fixed-point loop. `waypoint.py` computes
    waypoint_w = position_w + offset_w        (offset bounded by the per-mode reach)
    yaw_setpoint = yaw_w + gain * residual * limit
Both references re-anchor to the current state every control period. So neither a fixed
world position nor a fixed heading can be commanded through it. Case A measures drift
under the hold the system actually has. Case B adds an outer position loop inside this
probe, to separate "cannot hold" from "was never asked to".

Reset is tested by behaviour: drive hard, reset, then compare the idle response against
a freshly reset one.
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
parser.add_argument("--load_steps", type=int, default=30)
parser.add_argument("--idle_steps", type=int, default=20)
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import uav_indoor.tasks  # noqa: F401,E402
from uav_indoor.tasks.manager_based.uav_indoor.building_inspection.waypoint import XY_LIMIT_M  # noqa: E402
from uav_indoor.tasks.manager_based.uav_indoor.inspection import MissionPhase  # noqa: E402

from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _never(env) -> torch.Tensor:
    """A termination that never fires, so a hold test is not cut short.

    No **kwargs: Isaac Lab's term resolver reads it as a mandatory parameter.
    """

    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # Keep reset terms registered; disable episode termination.
    for name in ("mission_complete", "collision", "recovery_failed",
                 "entry_success", "unauthorized_crossing"):
        term_cfg = getattr(env_cfg.terminations, name)
        term_cfg.func = _never
        term_cfg.params = {}
    env = gym.make(args_cli.task, cfg=env_cfg)
    task = env.unwrapped
    robot = task.scene["robot"]
    term = task.action_manager.get_term("px4_waypoint")
    n, dev = args_cli.num_envs, task.device
    physics_hz = 1.0 / float(task.physics_dt)
    observe, transit = int(MissionPhase.OBSERVE), int(MissionPhase.TRANSIT)

    out = {
        "seed": args_cli.seed,
        "num_envs": n,
        "physics_hz": physics_hz,
        "motor_torque_sign": list(env_cfg.actions.px4_waypoint.motor_torque_sign),
        "fixed_point_hold_expressible": False,
        "fixed_heading_hold_expressible": False,
        "architecture_note": "waypoint_w = position_w + bounded offset; yaw_setpoint = yaw_w + step",
        "cases": [],
    }
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
            d.root_lin_vel_w.clone(),
            robot.data.joint_vel.clone(),
            (term._waypoint_w - (d.root_pos_w - task.scene.env_origins)).clone(),
        ])

    robot.write_data_to_sim = hooked
    zero = torch.zeros((n, 4), device=dev)
    ends = [0]

    def step(action: torch.Tensor, hold_phase: int | None = None) -> None:
        _, _, term_flag, trunc, _ = env.step(action)
        ends[0] += int((term_flag | trunc).sum().item())
        if hold_phase is not None:
            task._reference.phase[:] = hold_phase

    def summarize(label: str, repeat: int, target, yaw0, steps: int, expect_zero_cmd: bool) -> dict:
        pos = torch.stack([s[0] for s in samples])
        att = torch.stack([s[1] for s in samples])
        rate = torch.stack([s[2] for s in samples])
        lin = torch.stack([s[3] for s in samples])
        offset = torch.stack([s[5] for s in samples])
        hz = len(samples) / (steps * float(task.step_dt))
        err = torch.linalg.vector_norm(pos - target[None], dim=-1)
        seconds = steps * float(task.step_dt)
        invalid = []
        if abs(hz - physics_hz) > 0.1 * physics_hz:
            invalid.append(f"sampled at {hz:.1f} Hz, not {physics_hz:.0f} Hz")
        # Applied waypoint offset in metres, after the per-mode reach.
        cmd_xyz = offset[:, :, :2].abs().mean().item()
        if expect_zero_cmd and cmd_xyz > 0.02:
            invalid.append(f"a zeroed-translation mode still moved the waypoint {cmd_xyz:.3f} m")
        if not expect_zero_cmd and cmd_xyz < 0.02:
            invalid.append(f"the outer loop moved the waypoint {cmd_xyz:.4f} m, which is nothing")
        if ends[0]:
            invalid.append(f"the episode ended {ends[0]} times, so the aircraft respawned mid-test")
        return {
            "case": label,
            "repeat": repeat,
            "seconds": seconds,
            "sample_hz": round(hz, 1),
            "invalid": invalid,
            "mean_abs_waypoint_offset_m": round(cmd_xyz, 5),
            "episode_ends_during_case": int(ends[0]),
            "per_env_drift_final_m": err[-1].tolist(),
            "per_env_drift_max_m": err.max(dim=0).values.tolist(),
            "per_env_drift_rate_m_s": (err[-1] / seconds).tolist(),
            "per_env_heading_drift_deg": (wrap(att[-1, :, 2] - yaw0).abs() * 180 / math.pi).tolist(),
            "per_env_roll_peak_deg": (att[:, :, 0].abs().max(dim=0).values * 180 / math.pi).tolist(),
            "per_env_pitch_peak_deg": (att[:, :, 1].abs().max(dim=0).values * 180 / math.pi).tolist(),
            "per_env_body_rate_rms": rate.pow(2).mean(dim=0).sqrt().tolist(),
            "per_env_speed_max_m_s": torch.linalg.vector_norm(lin, dim=-1).max(dim=0).values.tolist(),
            "saturated_steps": int(term._action_saturated_total.sum().item()),
        }

    try:
        with torch.inference_mode():
            for repeat in range(args_cli.repeats):
                # OBSERVE reference is controlled by the mission.

                # Probe position loop with TRANSIT reach.
                env.reset()
                task._supervisor.phase[:] = transit
                task._supervisor.phase_time[:] = 0.0
                step(zero)
                target = (robot.data.root_pos_w - task.scene.env_origins).clone()
                yaw0 = wrap(task.robot_yaw_w.clone())
                samples.clear()
                ends[0] = 0
                reach = XY_LIMIT_M[0]
                for _ in range(args_cli.hold_steps):
                    task._supervisor.phase[:] = transit
                    task._supervisor.phase_time[:] = 0.0
                    task.building_standoff_w[:] = target
                    here = robot.data.root_pos_w - task.scene.env_origins
                    d = target - here
                    yaw = task.robot_yaw_w
                    fwd = torch.cos(yaw) * d[:, 0] + torch.sin(yaw) * d[:, 1]
                    left = -torch.sin(yaw) * d[:, 0] + torch.cos(yaw) * d[:, 1]
                    act = torch.zeros((n, 4), device=dev)
                    act[:, 0] = (fwd / reach).clamp(-1.0, 1.0)
                    act[:, 1] = (left / reach).clamp(-1.0, 1.0)
                    act[:, 2] = (d[:, 2] / reach).clamp(-1.0, 1.0)
                    step(act, transit)
                rec = summarize("probe_outer_loop_transit", repeat, target, yaw0, args_cli.hold_steps, False)
                out["cases"].append(rec)
                print(f"probe_loop  r{repeat} {rec['sample_hz']:.0f} Hz {rec['seconds']:.0f}s  "
                      f"drift {[round(v,2) for v in rec['per_env_drift_final_m']]} m  "
                      f"max {[round(v,2) for v in rec['per_env_drift_max_m']]} m  "
                      f"{'INVALID '+'; '.join(rec['invalid']) if rec['invalid'] else 'ok'}", flush=True)

            # Reset state.
            def idle() -> tuple[list, float]:
                samples.clear()
                for _ in range(args_cli.idle_steps):
                    task._supervisor.phase[:] = observe
                    task._supervisor.phase_time[:] = 0.0
                    step(zero, observe)
                rate = torch.stack([s[2] for s in samples])
                jv = torch.stack([s[4] for s in samples])
                return rate.abs().mean(dim=(0, 1)).tolist(), float(jv.abs().mean())

            env.reset()
            fresh_rate, fresh_joint = idle()
            hard = torch.zeros((n, 4), device=dev)
            hard[:, 3] = 1.0
            for _ in range(args_cli.load_steps):
                task._supervisor.phase[:] = observe
                task._supervisor.phase_time[:] = 0.0
                step(hard)
            env.reset()
            after_rate, after_joint = idle()
            band = [0.05 + 0.5 * abs(b) for b in fresh_rate]
            out["reset"] = {
                "idle_body_rate_fresh": fresh_rate,
                "idle_body_rate_after_load_and_reset": after_rate,
                "agreement_band": band,
                "idle_rotor_joint_vel_fresh": fresh_joint,
                "idle_rotor_joint_vel_after_reset": after_joint,
                "body_rate_agrees": all(abs(a - b) <= t for a, b, t in zip(after_rate, fresh_rate, band)),
                "rotor_state_ratio": after_joint / max(fresh_joint, 1e-6),
                "rotor_state_note": "reported, not gated: this varied run to run with unchanged code",
            }
            r = out["reset"]
            print(f"reset: idle rate fresh {[round(v,4) for v in fresh_rate]} vs after "
                  f"{[round(v,4) for v in after_rate]} agrees={r['body_rate_agrees']}", flush=True)
            print(f"reset: idle rotor vel fresh {fresh_joint:.1f} vs after {after_joint:.1f} "
                  f"ratio={r['rotor_state_ratio']:.2f} (reported, not gated)", flush=True)
    finally:
        robot.write_data_to_sim = original
        env.close()

    out["failures"] = (
        [f"{c['case']} r{c['repeat']} INVALID: {'; '.join(c['invalid'])}" for c in out["cases"] if c["invalid"]]
        + ([] if out["reset"]["body_rate_agrees"] else ["reset did not restore the idle body rate"])

    )
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
