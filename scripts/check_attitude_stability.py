#!/usr/bin/env python3
"""Isolate the heading oscillation in hover and straight flight, with no mission at all.

A full mission cannot diagnose basic attitude behaviour: every phase changes the reference,
the shield can veto an axis, and the survey keeps moving the aircraft between stations. Four
scripted regimes place the fault, and each one names what it rules out:

- ``hover``: the mission phase is held at OBSERVE, whose reference IS the current position
  and whose heading reference is the current heading when no person is tracked. A genuine
  position hold under a zero heading command: oscillation here is the attitude loop or the
  allocation, with nothing else acting.
- ``turn``: the same hold, with a constant heading command. The only regime that asks the
  outer heading loop for anything.
- ``straight``: the phase is held at TRANSIT with the target far ahead on the start heading,
  so the baseline is steady straight flight. Oscillation that appears only here is roll and
  pitch coupling into heading, not a heading-loop fault.
- ``diagonal``: the same, with the target far ahead diagonally. The per-mode reach bounds each
  horizontal axis separately, so a diagonal command is larger by root two; this says whether
  the speed ceiling and the oscillation are coupled.

The action is a RESIDUAL on the mission reference, so a zero action means "fly the reference",
not "hold still". An earlier version of this script left the mission in TRANSIT and called a
zero action hover; all four regimes then measured the same transit flight and agreed to within
8 percent, which is what gave it away.

Reported per regime: mean and peak heading rate, heading-rate reversals per minute, mean and
peak roll and pitch rate, and what heading rate was commanded. The last separates a loop that
faithfully tracks a bad command from one that cannot track a good one.
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
parser.add_argument("--seconds", type=float, default=20.0, help="Measured seconds per regime.")
parser.add_argument("--settle_s", type=float, default=4.0, help="Discarded seconds at each regime start.")
parser.add_argument("--turn_rate_frac", type=float, default=0.5, help="Heading command as a fraction of the limit.")
parser.add_argument("--yaw_gain", type=float, default=1.0)
# External actuator lag.
parser.add_argument("--motor_tau", type=float, default=None)
# Rotor animation.
parser.add_argument("--no_spin_joints", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--result", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Rendered sensor inputs.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import uav_indoor.tasks  # noqa: F401
from uav_indoor.tasks.manager_based.uav_indoor.inspection import MissionPhase

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Fixed phase and target regimes.
REGIMES = {
    "hover": {"phase": "OBSERVE", "action": (0.0, 0.0, 0.0, 0.0)},
    "turn": {"phase": "OBSERVE", "action": (0.0, 0.0, 0.0, None)},
    "straight": {"phase": "TRANSIT", "action": (0.0, 0.0, 0.0, 0.0), "target_b": (1.0, 0.0)},
    "diagonal": {"phase": "TRANSIT", "action": (0.0, 0.0, 0.0, 0.0), "target_b": (1.0, 1.0)},
}
# Straight-flight target distance.
TARGET_RANGE_M = 60.0

DEGREES = 180.0 / math.pi
# Heading-reversal threshold.
DEADBAND = 0.05


def summarize(name: str, samples: dict[str, list[float]], reversals: int, seconds: float, commanded: float) -> dict:
    def stats(key: str) -> tuple[float, float]:
        values = samples[key]
        if not values:
            return float("nan"), float("nan")
        return sum(values) / len(values), max(values)

    yaw_mean, yaw_peak = stats("yaw_rate")
    roll_mean, roll_peak = stats("roll_rate")
    pitch_mean, pitch_peak = stats("pitch_rate")
    return {
        "regime": name,
        "seconds": seconds,
        "samples": len(samples["yaw_rate"]),
        "yaw_rate_mean_deg_s": yaw_mean * DEGREES,
        "yaw_rate_peak_deg_s": yaw_peak * DEGREES,
        "yaw_reversals_per_min": reversals / (seconds / 60.0) if seconds > 0 else float("nan"),
        "roll_rate_mean_deg_s": roll_mean * DEGREES,
        "roll_rate_peak_deg_s": roll_peak * DEGREES,
        "pitch_rate_mean_deg_s": pitch_mean * DEGREES,
        "pitch_rate_peak_deg_s": pitch_peak * DEGREES,
        "commanded_yaw_rate_deg_s": commanded * DEGREES,
    }


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.yaw_gain != 1.0:
        raise SystemExit(
            "--yaw_gain no longer applies: the position action sends the yaw target "
            "to PX4 directly, with no heading-loop damping to set"
        )
    if args_cli.motor_tau is not None:
        env_cfg.actions.px4_waypoint.motor_tau = args_cli.motor_tau
    if args_cli.no_spin_joints:
        env_cfg.actions.px4_waypoint.visual_spin_joints = False
    # Single episode for all regimes.
    env_cfg.episode_length_s = (args_cli.seconds + args_cli.settle_s) * (len(REGIMES) + 1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    task = env.unwrapped
    env.reset()
    task.episode_records.clear()

    step_dt = float(task.step_dt)
    settle_steps = int(args_cli.settle_s / step_dt)
    measure_steps = int(args_cli.seconds / step_dt)
    yaw_rate_max = float(env_cfg.actions.px4_waypoint.yaw_rate_max)

    results = []
    try:
        for name, regime in REGIMES.items():
            values = list(regime["action"])
            if values[3] is None:
                values[3] = args_cli.turn_rate_frac
            action = torch.tensor(values, device=task.device).repeat(args_cli.num_envs, 1)
            commanded = abs(float(values[3])) * yaw_rate_max
            phase = int(getattr(MissionPhase, regime["phase"]))
            target_b = regime.get("target_b")

            samples = {"yaw_rate": [], "roll_rate": [], "pitch_rate": []}
            reversals = 0
            previous_rate = None
            previous_yaw = None
            for step in range(settle_steps + measure_steps):
                with torch.inference_mode():
                    # Hold phase and phase clock.
                    task._supervisor.phase[:] = phase
                    task._supervisor.phase_time[:] = 0.0
                    if target_b is not None:
                        here = task.scene["robot"].data.root_pos_w - task.scene.env_origins
                        heading = task.robot_yaw_w
                        forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
                        left = torch.stack((-torch.sin(heading), torch.cos(heading)), dim=-1)
                        offset = TARGET_RANGE_M * (target_b[0] * forward + target_b[1] * left)
                        task.building_standoff_w[:, :2] = here[:, :2] + offset
                        task.building_standoff_w[:, 2] = here[:, 2]
                    env.step(action)
                    yaw = task.robot_yaw_w.clone()
                    body_rate = task.scene["robot"].data.root_ang_vel_b.clone()
                    if previous_yaw is not None:
                        delta = torch.atan2(torch.sin(yaw - previous_yaw), torch.cos(yaw - previous_yaw))
                        rate = delta / step_dt
                        if step >= settle_steps:
                            samples["yaw_rate"].append(float(rate.abs().mean()))
                            samples["roll_rate"].append(float(body_rate[:, 0].abs().mean()))
                            samples["pitch_rate"].append(float(body_rate[:, 1].abs().mean()))
                            if previous_rate is not None:
                                flipped = (
                                    (rate.abs() > DEADBAND)
                                    & (previous_rate.abs() > DEADBAND)
                                    & (torch.sign(rate) != torch.sign(previous_rate))
                                )
                                # Count simultaneous heading reversals.
                                reversals += int(float(flipped.float().mean()) > 0.5)
                            previous_rate = rate
                    previous_yaw = yaw
            summary = summarize(name, samples, reversals, args_cli.seconds, commanded)
            results.append(summary)
            print(
                f"ATTITUDE {name}: yaw_mean={summary['yaw_rate_mean_deg_s']:.2f} deg/s "
                f"peak={summary['yaw_rate_peak_deg_s']:.1f} "
                f"reversals_per_min={summary['yaw_reversals_per_min']:.1f} "
                f"commanded={summary['commanded_yaw_rate_deg_s']:.2f} deg/s "
                f"roll_mean={summary['roll_rate_mean_deg_s']:.2f} "
                f"pitch_mean={summary['pitch_rate_mean_deg_s']:.2f}",
                flush=True,
            )
    finally:
        env.close()

    hover = next((r for r in results if r["regime"] == "hover"), None)
    verdict = "UNKNOWN"
    reading = "no hover measurement, so nothing is isolated"
    if hover is not None:
        if hover["yaw_reversals_per_min"] > 60.0:
            verdict = "PLANT"
            reading = (
                "the aircraft oscillates in hover under a zero heading command, so the fault "
                "is in the controller or the actuator path, not in mission or waypoint logic. "
                "It is not yaw specific: roll and pitch rates are as large or larger"
            )
        elif hover["yaw_rate_mean_deg_s"] < 5.0:
            verdict = "NOT_PLANT"
            reading = (
                "hover is steady, so the mission oscillation comes from what commands the "
                "heading, not from the attitude loop"
            )
        else:
            verdict = "PARTIAL"
            reading = "hover drifts without a clear limit cycle; the other regimes decide"
    print(f"ATTITUDE_VERDICT {verdict}: {reading}", flush=True)

    if args_cli.result:
        Path(args_cli.result).write_text(
            json.dumps(
                {
                    "gate": "attitude_stability",
                    # Diagnostic result.
                    "verdict": verdict,
                    "reading": reading,
                    "failures": [],
                    "yaw_gain": args_cli.yaw_gain,
                    "motor_tau": args_cli.motor_tau,
                    "no_spin_joints": args_cli.no_spin_joints,
                    "seconds_per_regime": args_cli.seconds,
                    "settle_s": args_cli.settle_s,
                    "regimes": results,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    status = main()
    simulation_app.close()
    raise SystemExit(status)
