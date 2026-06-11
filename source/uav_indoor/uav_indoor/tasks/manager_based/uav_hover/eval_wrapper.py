from __future__ import annotations

import gymnasium as gym
import torch

from .mdp.events import get_episode_xy_ref, get_hover_target


class HoverEvalWrapper(gym.Wrapper):
    """Logs weight-invariant hover metrics into extras["log"] for rsl_rl/wandb.

    Keys contain "/" so rsl_rl logs them verbatim, averaged over the rollout:
      Eval/hover_success  - fraction of steps within tol of BOTH commanded height and spawn xy
      Eval/height_err_m   - mean |commanded height - current height| (m)
      Eval/xy_err_m       - mean ||spawn xy - current xy|| (m)
    Runs after the inner step, so the injected keys are not clobbered by _reset_idx.
    """

    def __init__(self, env: gym.Env, height_tol: float = 0.25, xy_tol: float = 0.25) -> None:
        super().__init__(env)
        self.height_tol = height_tol
        self.xy_tol = xy_tol

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        u = self.env.unwrapped
        asset = u.scene["robot"]
        z = asset.data.root_pos_w[:, 2] - u.scene.env_origins[:, 2]
        height_err = torch.abs(get_hover_target(u) - z)
        xy = asset.data.root_pos_w[:, :2] - u.scene.env_origins[:, :2]
        xy_err = torch.norm(get_episode_xy_ref(u) - xy, dim=-1)
        success = ((height_err < self.height_tol) & (xy_err < self.xy_tol)).float()

        log = info.setdefault("log", {})
        log["Eval/hover_success"] = success
        log["Eval/height_err_m"] = height_err
        log["Eval/xy_err_m"] = xy_err
        return obs, reward, terminated, truncated, info
