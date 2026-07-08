# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a deterministic hover controller through the UAV hover environment.

This is a diagnostic baseline, not a learned policy. If this script cannot hold
height and XY with the PX4 velocity action term, debug the simulator/controller
before spending more PPO time.
"""

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Deterministic hover baseline for the UAV hover task.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Template-Uav-Hover-v0")
parser.add_argument("--steps", type=int, default=1800)
parser.add_argument("--log_interval", type=int, default=60)
parser.add_argument("--kp_xy", type=float, default=0.8)
parser.add_argument("--kp_z", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi

import uav_indoor.tasks  # noqa: F401
from uav_indoor.tasks.manager_based.uav_hover.mdp.events import (
    get_episode_xy_ref,
    get_episode_yaw_ref,
    get_hover_target,
)


def _make_actions(env, kp_xy: float, kp_z: float) -> torch.Tensor:
    """Compute normalized body-frame velocity + absolute-yaw actions."""
    unwrapped = env.unwrapped
    action_term = unwrapped.action_manager.get_term("px4_velocity")
    asset = unwrapped.scene["robot"]

    pos = asset.data.root_pos_w - unwrapped.scene.env_origins
    xy_err = get_episode_xy_ref(unwrapped) - pos[:, :2]
    z_err = get_hover_target(unwrapped) - pos[:, 2]

    vel_w = torch.zeros(unwrapped.num_envs, 3, device=unwrapped.device)
    vel_w[:, :2] = torch.clamp(kp_xy * xy_err, -action_term.cfg.v_max_xy, action_term.cfg.v_max_xy)
    vel_w[:, 2] = torch.clamp(kp_z * z_err, -action_term.cfg.v_max_z, action_term.cfg.v_max_z)

    vel_b = quat_apply_inverse(asset.data.root_quat_w, vel_w)

    actions = torch.zeros(unwrapped.num_envs, 4, device=unwrapped.device)
    actions[:, 0] = torch.clamp(vel_b[:, 0] / action_term.cfg.v_max_xy, -1.0, 1.0)
    actions[:, 1] = torch.clamp(vel_b[:, 1] / action_term.cfg.v_max_xy, -1.0, 1.0)
    actions[:, 2] = torch.clamp(vel_b[:, 2] / action_term.cfg.v_max_z, -1.0, 1.0)

    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    yaw_err = wrap_to_pi(get_episode_yaw_ref(unwrapped) - yaw)
    yaw_cmd = wrap_to_pi(yaw + yaw_err)
    actions[:, 3] = torch.clamp(yaw_cmd / math.pi, -1.0, 1.0)
    return actions


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    with torch.inference_mode():
        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                break

            actions = _make_actions(env, args_cli.kp_xy, args_cli.kp_z)
            _, _, _, _, extras = env.step(actions)

            if step % args_cli.log_interval == 0:
                log = extras.get("log", {})
                height_err = log.get("Eval/height_err_m")
                xy_err = log.get("Eval/xy_err_m")
                success = log.get("Eval/hover_success")
                print(
                    "[HOVER] "
                    f"step={step:05d} "
                    f"height_err={height_err.item():.3f} "
                    f"xy_err={xy_err.item():.3f} "
                    f"success={success.item():.3f}"
                )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
