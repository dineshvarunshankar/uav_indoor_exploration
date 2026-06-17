from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv

from .mdp.events import get_episode_xy_ref, get_hover_target


class UavHoverEnv(ManagerBasedRLEnv):
    """Hover env with episode-end station-keeping eval for rsl_rl/wandb sweeps.

    On timeout (t=60s), logs pre-reset metrics into extras["log"]:
      Eval/hover_success - within tol of commanded height AND spawn xy
      Eval/height_err_m  - |commanded height - current height| (m)
      Eval/xy_err_m      - ||spawn xy - current xy|| (m)
    """

    height_tol: float = 0.25
    xy_tol: float = 0.25

    def step(self, action):
        log = self.extras.get("log")
        if isinstance(log, dict):
            for key in [k for k in log if k.startswith("Eval/")]:
                del log[key]
        return super().step(action)

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        if env_ids is None:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        # reset_time_outs is only populated after the first step; skip eval on initial reset
        if hasattr(self, "reset_time_outs"):
            timeout_ids = env_ids_t[self.reset_time_outs[env_ids_t]]
            eval_log = self._station_keeping_eval(timeout_ids) if timeout_ids.numel() > 0 else {}
        else:
            eval_log = {}

        super()._reset_idx(env_ids)

        if eval_log:
            self.extras.setdefault("log", {}).update(eval_log)

    def _station_keeping_eval(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        asset = self.scene["robot"]
        z = asset.data.root_pos_w[env_ids, 2] - self.scene.env_origins[env_ids, 2]
        height_err = torch.abs(get_hover_target(self)[env_ids] - z)
        xy = asset.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2]
        xy_err = torch.norm(get_episode_xy_ref(self)[env_ids] - xy, dim=-1)
        success = ((height_err < self.height_tol) & (xy_err < self.xy_tol)).float()
        return {
            "Eval/hover_success": success,
            "Eval/height_err_m": height_err,
            "Eval/xy_err_m": xy_err,
        }
