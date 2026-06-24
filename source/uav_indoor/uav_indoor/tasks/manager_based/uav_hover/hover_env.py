from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from .mdp.events import get_episode_xy_ref, get_hover_target


class UavHoverEnv(ManagerBasedRLEnv):
    """Hover env that logs station-keeping eval metrics every step for rsl_rl/wandb.

    Adds to extras["log"], averaged over envs:
      Eval/hover_success - fraction within height_tol AND xy_tol
      Eval/height_err_m  - mean |commanded height - current height| (m)
      Eval/xy_err_m      - mean ||spawn xy - current xy|| (m)
    """

    height_tol: float = 0.25
    xy_tol: float = 0.25

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
        success = ((height_err < self.height_tol) & (xy_err < self.xy_tol)).float()
        return {
            "Eval/hover_success": success.mean(),
            "Eval/height_err_m": height_err.mean(),
            "Eval/xy_err_m": xy_err.mean(),
        }
