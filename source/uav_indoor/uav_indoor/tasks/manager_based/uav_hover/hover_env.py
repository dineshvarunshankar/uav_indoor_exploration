from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from .mdp.events import get_episode_xy_ref, get_episode_yaw_ref, get_hover_target


class UavHoverEnv(ManagerBasedRLEnv):
    """Hover env that logs station-keeping eval metrics every step for rsl_rl/wandb."""

    height_tol: float = 0.25
    xy_tol: float = 0.25
    yaw_tol: float = 0.15

    def step(self, action):
        obs, reward, terminated, truncated, extras = super().step(action)
        extras.setdefault("log", {}).update(self._station_keeping_eval())
        return obs, reward, terminated, truncated, extras

    def _station_keeping_eval(self) -> dict[str, torch.Tensor]:
        asset = self.scene["robot"]
        z = asset.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        height_err = torch.abs(get_hover_target(self) - z)
        xy = asset.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        xy_err = torch.norm(get_episode_xy_ref(self) - xy, dim=-1)
        _, _, yaw_now = euler_xyz_from_quat(asset.data.root_quat_w)
        yaw_err = torch.abs(wrap_to_pi(get_episode_yaw_ref(self) - wrap_to_pi(yaw_now)))
        success = (
            (height_err < self.height_tol)
            & (xy_err < self.xy_tol)
            & (yaw_err < self.yaw_tol)
        ).float()
        return {
            "Eval/hover_success": success.mean(),
            "Eval/height_err_m": height_err.mean(),
            "Eval/xy_err_m": xy_err.mean(),
            "Eval/yaw_err_rad": yaw_err.mean(),
        }
