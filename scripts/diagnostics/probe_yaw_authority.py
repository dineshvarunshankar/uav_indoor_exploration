#!/usr/bin/env python3
"""Where does the yaw reaction torque go? Read-only.

An open-loop rotor-frame z torque of 0.0369 Nm produced 1.5 percent of the yaw
acceleration the measured inertia predicts, while roll and pitch matched within
2 percent. The rotors are separate articulation links on joints with zero stiffness
and zero damping, so a torque about a rotor's own spin axis may spin the rotor
instead of turning the airframe.

Applies the identical torque to the rotor links and to the body link and compares.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Template-Uav-Building-Inspection-v0")
parser.add_argument("--burst_steps", type=int, default=24)
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
TORQUE_Z_NM = 0.03684  # the value the earlier open-loop yaw pattern applied


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    task = env.unwrapped
    env.reset()
    robot = task.scene["robot"]
    term = task.action_manager.get_term("px4_waypoint")
    dt = task.physics_dt
    dev = task.device
    origin = task.scene.env_origins[0]

    izz = robot.root_physx_view.get_inertias()[0][0].tolist()[8]
    body_id = robot.body_names.index("body")
    prop_ids = [int(i) for i in term._prop_body_ids]

    out = {
        "izz": izz,
        "applied_torque_z_Nm": TORQUE_Z_NM,
        "predicted_ang_accel_z": TORQUE_Z_NM / izz,
        "body_names": list(robot.body_names),
        "joint_names": list(robot.joint_names),
        "prop_body_ids": prop_ids,
        "body_body_id": body_id,
    }
    view = robot.root_physx_view
    for name, getter in (
        ("joint_stiffness", "get_dof_stiffnesses"),
        ("joint_damping", "get_dof_dampings"),
        ("joint_max_force", "get_dof_max_forces"),
    ):
        try:
            out[name] = getattr(view, getter)()[0].tolist()
        except Exception as exc:  # noqa: BLE001
            out[name] = f"unavailable: {exc!r}"
    try:
        out["joint_limits"] = view.get_dof_limits()[0].tolist()
    except Exception as exc:  # noqa: BLE001
        out["joint_limits"] = f"unavailable: {exc!r}"

    def park() -> None:
        pose = torch.zeros((1, 7), device=dev)
        pose[0, :3] = torch.tensor(FREE_AIR, device=dev) + origin
        pose[0, 3] = 1.0
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
        robot.write_joint_state_to_sim(
            torch.zeros((1, robot.num_joints), device=dev),
            torch.zeros((1, robot.num_joints), device=dev),
        )
        task.sim.step(render=False)
        task.scene.update(dt=dt)

    def burst(target_ids: list[int], per_body_torque: float) -> dict:
        park()
        n = len(target_ids)
        forces = torch.zeros((1, n, 3), device=dev)
        torques = torch.zeros((1, n, 3), device=dev)
        torques[0, :, 2] = per_body_torque
        w0 = robot.data.root_ang_vel_b[0].clone()
        j0 = robot.data.joint_vel[0].clone()
        for _ in range(args_cli.burst_steps):
            robot.instantaneous_wrench_composer.set_forces_and_torques(
                forces=forces, torques=torques, body_ids=target_ids, is_global=False
            )
            robot.write_data_to_sim()
            task.sim.step(render=False)
            task.scene.update(dt=dt)
        w1 = robot.data.root_ang_vel_b[0].clone()
        j1 = robot.data.joint_vel[0].clone()
        span = args_cli.burst_steps * dt
        return {
            "ang_accel_body": ((w1 - w0) / span).tolist(),
            "joint_accel": ((j1 - j0) / span).tolist(),
            "final_joint_vel": j1.tolist(),
        }

    cases = {
        "torque_on_rotor_links": (prop_ids, TORQUE_Z_NM / len(prop_ids)),
        "torque_on_body_link": ([body_id], TORQUE_Z_NM),
    }
    out["cases"] = {}
    for name, (ids, per) in cases.items():
        result = burst(ids, per)
        out["cases"][name] = result | {"body_ids": ids, "per_body_torque_Nm": per}
        az = result["ang_accel_body"][2]
        print(f"{name:26s} yaw accel {az:9.3f} rad/s^2  "
              f"({az / (TORQUE_Z_NM / izz) * 100:6.1f}% of predicted)  "
              f"joint accel {[round(v, 1) for v in result['joint_accel']]}", flush=True)

    Path(args_cli.out).write_text(json.dumps(out, indent=2))
    print(f"PROBE_WROTE {args_cli.out}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    status = main()
    simulation_app.close()
    raise SystemExit(status)
