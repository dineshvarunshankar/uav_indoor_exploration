#!/usr/bin/env python3
"""Does a commanded torque turn the aircraft the way it was asked? Read-only.

Two parts, neither of which uses the mission, a policy, or a reward:

A. Open loop. Write a known rotor wrench straight to the articulation, bypassing the
   controller, and compare the measured angular acceleration against torque / inertia.
   This checks frames, signs and magnitudes on the physics path.

B. Closed loop. Command a yaw step through the action term and log the setpoint the
   controller actually produced beside the measured heading. This checks whether the
   whole chain turns the aircraft the way it was asked.

`mdp/actions.py` uses motor_torque_sign (+1, +1, -1, -1). Px4Mixer.hpp allocates with
_rotor_yaw_sign (-1, -1, +1, +1) and states the two must agree. They are negations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Uav-Building-Inspection-v0")
parser.add_argument("--burst_steps", type=int, default=24, help="Physics steps per open-loop burst.")
parser.add_argument("--yaw_steps", type=int, default=40, help="Env steps for the closed-loop yaw step.")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import uav_indoor.tasks  # noqa: F401,E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

FREE_AIR = (-40.0, -30.0, 30.0)


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    task = env.unwrapped
    env.reset()

    robot = task.scene["robot"]
    term = task.action_manager.get_term("px4_waypoint")
    cfg = term.cfg
    dt = task.physics_dt
    origin = task.scene.env_origins[0]
    dev = task.device

    inertia = robot.root_physx_view.get_inertias()[0][0].tolist()
    ixx, iyy, izz = inertia[0], inertia[4], inertia[8]
    mass = float(robot.root_physx_view.get_masses()[0].sum())
    body_order = list(robot.body_names)
    prop_ids = term._prop_body_ids
    k_ratio = abs(cfg.k_torque) / abs(cfg.k_thrust)
    torque_sign = torch.tensor(cfg.motor_torque_sign, device=dev)
    arm_py, arm_px = 0.088, 0.066  # CA_ROTOR_PY, CA_ROTOR_PX from Px4Mixer.hpp

    out: dict = {
        "mass_kg": mass,
        "inertia_diag": [ixx, iyy, izz],
        "body_names": body_order,
        "prop_body_ids": [int(i) for i in prop_ids],
        "motor_torque_sign": list(cfg.motor_torque_sign),
        "mixer_rotor_yaw_sign": [-1.0, -1.0, 1.0, 1.0],
        "k_torque_over_k_thrust_m": k_ratio,
        "open_loop": [],
    }

    def park() -> None:
        pose = torch.zeros((1, 7), device=dev)
        pose[0, :3] = torch.tensor(FREE_AIR, device=dev) + origin
        pose[0, 3] = 1.0
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
        task.sim.step(render=False)
        task.scene.update(dt=dt)

    # Open-loop torque controls.
    hover_rotor = mass * 9.81 / 4.0
    delta = 0.25 * hover_rotor
    patterns = {
        # Raise the pair the mixer calls CCW. Python's reaction sign says +z torque.
        "yaw_raise_rotor0_rotor1": (delta, delta, -delta, -delta),
        # rotor_py_sign = (-1, +1, +1, -1): raising 1 and 2 rolls one way.
        "roll_raise_rotor1_rotor2": (-delta, delta, delta, -delta),
        # rotor_px_sign = (+1, -1, +1, -1): raising 0 and 2 pitches one way.
        "pitch_raise_rotor0_rotor2": (delta, -delta, delta, -delta),
    }
    for name, thrusts in patterns.items():
        park()
        t = torch.tensor(thrusts, device=dev)
        forces = torch.zeros((1, len(prop_ids), 3), device=dev)
        torques = torch.zeros((1, len(prop_ids), 3), device=dev)
        forces[0, :, 2] = t
        torques[0, :, 2] = torque_sign * k_ratio * t
        start = robot.data.root_ang_vel_b[0].clone()
        for _ in range(args_cli.burst_steps):
            robot.instantaneous_wrench_composer.set_forces_and_torques(
                forces=forces, torques=torques, body_ids=prop_ids, is_global=False
            )
            # Write the force buffer to PhysX.
            robot.write_data_to_sim()
            task.sim.step(render=False)
            task.scene.update(dt=dt)
        end = robot.data.root_ang_vel_b[0].clone()
        measured = ((end - start) / (args_cli.burst_steps * dt)).tolist()
        # Predicted body torque from the same numbers the code uses.
        tx = float(sum(s * f for s, f in zip((-1.0, 1.0, 1.0, -1.0), thrusts)) * arm_py)
        ty = float(sum(s * f for s, f in zip((1.0, -1.0, 1.0, -1.0), thrusts)) * arm_px)
        tz = float((torque_sign.cpu() * k_ratio * torch.tensor(thrusts).cpu()).sum())
        out["open_loop"].append({
            "pattern": name,
            "thrust_per_rotor_N": list(thrusts),
            "predicted_torque_Nm": [tx, ty, tz],
            "predicted_ang_accel_rad_s2": [tx / ixx, ty / iyy, tz / izz],
            "measured_ang_accel_rad_s2": measured,
        })
        print(f"{name:28s} predicted {[round(v,2) for v in (tx/ixx, ty/iyy, tz/izz)]} "
              f"measured {[round(v,2) for v in measured]}", flush=True)

    # Closed-loop yaw step.
    env.reset()
    action = torch.zeros((1, 4), device=dev)
    action[0, 3] = 1.0  # full positive yaw residual
    trace = []
    for i in range(args_cli.yaw_steps):
        with torch.inference_mode():
            env.step(action)
        trace.append({
            "step": i,
            "yaw_setpoint_rad": float(term._yaw_setpoint[0].item()),
            "yaw_measured_rad": float(task.robot_yaw_w[0].item()),
            "yaw_rate_body_z": float(robot.data.root_ang_vel_b[0, 2].item()),
        })
    out["closed_loop_yaw_step"] = trace
    setpoint_delta = trace[-1]["yaw_setpoint_rad"] - trace[0]["yaw_setpoint_rad"]
    measured_delta = math.atan2(
        math.sin(trace[-1]["yaw_measured_rad"] - trace[0]["yaw_measured_rad"]),
        math.cos(trace[-1]["yaw_measured_rad"] - trace[0]["yaw_measured_rad"]),
    )
    mean_rate = sum(t["yaw_rate_body_z"] for t in trace) / len(trace)
    out["closed_loop_summary"] = {
        "commanded_residual_yaw": 1.0,
        "setpoint_delta_rad": setpoint_delta,
        "measured_yaw_delta_rad": measured_delta,
        "mean_body_yaw_rate_rad_s": mean_rate,
        "setpoint_and_response_agree_in_sign": (mean_rate > 0) == (setpoint_delta >= 0)
        if abs(mean_rate) > 1e-3 else None,
    }
    print(f"closed loop: setpoint moved {setpoint_delta:+.3f} rad, aircraft yawed "
          f"{measured_delta:+.3f} rad, mean body yaw rate {mean_rate:+.3f} rad/s", flush=True)

    Path(args_cli.out).write_text(json.dumps(out, indent=2))
    print(f"PROBE_WROTE {args_cli.out}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    status = main()
    simulation_app.close()
    raise SystemExit(status)
