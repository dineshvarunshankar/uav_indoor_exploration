"""Protected breadcrumb route for opening entry and exit.

The buffer records the corridor from the exterior checkpoint through the entry manoeuvre.
The aircraft leaves by flying the same crumbs in reverse while it faces forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

BREADCRUMB_FIELDS = 6  # position, yaw, clearance, and time


@dataclass(frozen=True)
class RecoveryCfg:
    """Route capacity and spacing.

    PROVISIONAL. 32 crumbs at 0.25 m cover about 8 m of corridor, which exceeds the
    0.75 m to 1.0 m entry depth by a wide margin.
    """

    max_breadcrumbs: int = 32
    min_spacing_m: float = 0.25
    reach_radius_m: float = 0.4


class BreadcrumbBuffer:
    """Batched entry route with a protected write path.

    Writes are accepted only while the buffer records. Sealing it blocks every later
    write, including an offboard update. A full buffer stops recording instead of
    overwriting its start, because the earliest crumbs are the way back outside.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cfg: RecoveryCfg | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or RecoveryCfg()
        crumbs = self.cfg.max_breadcrumbs
        self.max_breadcrumbs = crumbs

        self.position_w = torch.zeros(num_envs, crumbs, 3, device=self.device)
        self.yaw_w = torch.zeros(num_envs, crumbs, device=self.device)
        self.clearance_m = torch.zeros(num_envs, crumbs, device=self.device)
        self.time_s = torch.zeros(num_envs, crumbs, device=self.device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.exit_index = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.recording = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.overflow = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.failed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._slot_index = torch.arange(crumbs, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.position_w[env_ids] = 0.0
        self.yaw_w[env_ids] = 0.0
        self.clearance_m[env_ids] = 0.0
        self.time_s[env_ids] = 0.0
        self.count[env_ids] = 0
        self.exit_index[env_ids] = -1
        self.recording[env_ids] = False
        self.overflow[env_ids] = False
        self.failed[env_ids] = False

    def start(self, mask: torch.Tensor) -> None:
        """Open a new route at the exterior checkpoint."""

        keep = ~mask
        self.count = self.count * keep
        self.exit_index = torch.where(mask, torch.full_like(self.exit_index, -1), self.exit_index)
        self.overflow = self.overflow & keep
        self.failed = self.failed & keep
        self.recording = self.recording | mask

    def seal(self, mask: torch.Tensor) -> None:
        """Close the route and prepare the reverse traversal."""

        sealing = mask & self.recording
        self.recording = self.recording & ~mask
        self.exit_index = torch.where(sealing, self.count - 1, self.exit_index)

    def push(
        self,
        position_w: torch.Tensor,
        yaw_w: torch.Tensor,
        clearance_m: torch.Tensor,
        time_s: float,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Append one crumb where the aircraft has moved far enough since the last one."""

        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        last = self.position_w.gather(
            1, (self.count - 1).clamp_min(0)[:, None, None].expand(-1, -1, 3)
        ).squeeze(1)
        travelled = torch.linalg.vector_norm(position_w - last, dim=-1)
        spaced = (self.count == 0) | (travelled >= self.cfg.min_spacing_m)
        room = self.count < self.max_breadcrumbs
        self.overflow = self.overflow | (self.recording & mask & spaced & ~room)

        write = mask & self.recording & spaced & room
        slot = write[:, None] & (self._slot_index[None, :] == self.count[:, None])
        self.position_w = torch.where(slot[..., None], position_w[:, None, :], self.position_w)
        self.yaw_w = torch.where(slot, yaw_w[:, None], self.yaw_w)
        self.clearance_m = torch.where(slot, clearance_m[:, None], self.clearance_m)
        self.time_s = torch.where(slot, torch.full_like(self.time_s, time_s), self.time_s)
        self.count = self.count + write.long()

    def next_crumb(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the next exit crumb, newest first, and whether one remains."""

        valid = self.exit_index >= 0
        index = self.exit_index.clamp_min(0)
        position = self.position_w.gather(1, index[:, None, None].expand(-1, -1, 3)).squeeze(1)
        yaw = self.yaw_w.gather(1, index[:, None]).squeeze(1)
        return position, yaw, valid

    def advance(self, position_w: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Consume the current crumb once the aircraft reaches it. Returns the exit-complete mask."""

        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        target, _, valid = self.next_crumb()
        reached = mask & valid & (torch.linalg.vector_norm(position_w - target, dim=-1) <= self.cfg.reach_radius_m)
        self.exit_index = self.exit_index - reached.long()
        return self.exit_complete

    @property
    def exit_complete(self) -> torch.Tensor:
        """True when every crumb of a sealed route has been consumed."""

        return (self.exit_index < 0) & (self.count > 0) & ~self.recording

    def mark_failed(self, mask: torch.Tensor) -> None:
        """Record ``ENTRY_RECOVERY_FAILED`` for environments that cannot complete the exit."""

        self.failed = self.failed | mask

    def observation(self, robot_position_w: torch.Tensor) -> torch.Tensor:
        """Actor features: crumb validity, body-frame offset, and route progress."""

        target, _, valid = self.next_crumb()
        offset = torch.where(valid[:, None], target - robot_position_w, torch.zeros_like(target))
        remaining = (self.exit_index + 1).clamp_min(0).float() / self.max_breadcrumbs
        return torch.cat((valid.float()[:, None], offset, remaining[:, None]), dim=-1)
