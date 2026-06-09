from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# DIAGNOSTIC: drop-in replacement for mdp.illegal_contact that also records, per crash,
# the episode step at which it happened. Bins are printed to stdout periodically so we can
# see whether crashes cluster at takeoff (early steps) or are spread across the episode.
# Remove this term and switch the cfg back to mdp.illegal_contact once the question is answered.

# bin upper edges (in env steps); episode is ~1200 steps at 60 Hz
_CRASH_BINS = (10, 30, 100, 300, 600, 1200)


def illegal_contact_logged(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    report_every: int = 200,
) -> torch.Tensor:
    """Same as mdp.illegal_contact, but histograms the crash step and prints periodically."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    crashed = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1,
    )

    # lazily allocate persistent counters on the env
    if not hasattr(env, "_crash_step_hist"):
        env._crash_step_hist = torch.zeros(len(_CRASH_BINS), device=env.device, dtype=torch.long)
        env._crash_report_ctr = 0

    crash_ids = crashed.nonzero(as_tuple=False).squeeze(-1)
    if crash_ids.numel() > 0:
        steps = env.episode_length_buf[crash_ids]  # current step index of each crashing env
        bins = torch.tensor(_CRASH_BINS, device=env.device)
        idx = torch.bucketize(steps, bins, right=True).clamp_(max=len(_CRASH_BINS) - 1)
        env._crash_step_hist += torch.bincount(idx, minlength=len(_CRASH_BINS))

    env._crash_report_ctr += 1
    if env._crash_report_ctr % report_every == 0:
        hist = env._crash_step_hist
        total = int(hist.sum().item())
        if total > 0:
            edges = (0,) + _CRASH_BINS
            parts = [
                f"[{edges[i]}-{edges[i + 1]}): {int(hist[i].item())} ({100 * hist[i].item() / total:.0f}%)"
                for i in range(len(_CRASH_BINS))
            ]
            print(f"[CRASH-STEP HIST] n={total} | " + "  ".join(parts), flush=True)

    return crashed
