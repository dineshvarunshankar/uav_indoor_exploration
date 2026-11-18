"""Starling-derived deterministic takeoff contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import torch


@dataclass(frozen=True)
class TakeoffCfg:
    """Fixed vertical takeoff and arrival criteria."""

    height_m: float = 2.5
    acceptance_distance_m: float = 0.3
    acceptance_time_s: float = 1.0
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if self.height_m <= 0.0:
            raise ValueError("takeoff height must be positive")
        if self.acceptance_distance_m <= 0.0:
            raise ValueError("takeoff acceptance distance must be positive")
        if self.acceptance_time_s <= 0.0:
            raise ValueError("takeoff acceptance time must be positive")
        if self.timeout_s is not None and self.timeout_s <= 0.0:
            raise ValueError("takeoff timeout must be positive when configured")


class PremissionTakeoff:
    """Batched latched target and continuous-distance completion monitor."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cfg: TakeoffCfg | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or TakeoffCfg()
        self.target_position = torch.zeros(num_envs, 3, device=self.device)
        self.target_yaw = torch.zeros(num_envs, device=self.device)
        self.acceptance_time = torch.zeros(num_envs, device=self.device)
        self.elapsed_time = torch.zeros(num_envs, device=self.device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.complete = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.timed_out = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.handoff = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def _ids(self, env_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def start(
        self,
        env_ids: torch.Tensor | Sequence[int],
        launch_position: torch.Tensor,
        launch_yaw: torch.Tensor,
    ) -> None:
        """Latch one vertical target from each selected environment's launch pose."""

        ids = self._ids(env_ids)
        positions = torch.as_tensor(launch_position, dtype=torch.float32, device=self.device)
        yaws = torch.as_tensor(launch_yaw, dtype=torch.float32, device=self.device)
        if positions.shape == (self.num_envs, 3):
            positions = positions[ids]
        if yaws.shape == (self.num_envs,):
            yaws = yaws[ids]
        if positions.shape != (ids.numel(), 3) or yaws.shape != (ids.numel(),):
            raise ValueError("launch pose must provide one position and yaw per selected environment")
        if not torch.isfinite(positions).all() or not torch.isfinite(yaws).all():
            raise ValueError("launch pose must be finite")

        self.target_position[ids] = positions
        self.target_position[ids, 2] += self.cfg.height_m
        self.target_yaw[ids] = yaws
        self.acceptance_time[ids] = 0.0
        self.elapsed_time[ids] = 0.0
        self.active[ids] = True
        self.complete[ids] = False
        self.timed_out[ids] = False
        self.handoff[ids] = False

    def start_mask(
        self,
        mask: torch.Tensor,
        launch_position: torch.Tensor,
        launch_yaw: torch.Tensor,
    ) -> None:
        """Latch full-batch launch poses where ``mask`` is true without a host sync."""

        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        positions = torch.as_tensor(launch_position, dtype=torch.float32, device=self.device)
        yaws = torch.as_tensor(launch_yaw, dtype=torch.float32, device=self.device)
        if mask.shape != (self.num_envs,) or positions.shape != (self.num_envs, 3):
            raise ValueError("mask and launch positions must cover every environment")
        if yaws.shape != (self.num_envs,):
            raise ValueError("launch yaw must cover every environment")
        target = positions.clone()
        target[:, 2] += self.cfg.height_m
        self.target_position = torch.where(mask[:, None], target, self.target_position)
        self.target_yaw = torch.where(mask, yaws, self.target_yaw)
        self.acceptance_time = torch.where(mask, torch.zeros_like(self.acceptance_time), self.acceptance_time)
        self.elapsed_time = torch.where(mask, torch.zeros_like(self.elapsed_time), self.elapsed_time)
        self.active |= mask
        self.complete &= ~mask
        self.timed_out &= ~mask
        self.handoff &= ~mask

    def update(
        self,
        dt_s: float,
        *,
        position: torch.Tensor,
        state_fresh: torch.Tensor,
    ) -> None:
        """Advance arrival dwell and timeout from measured state."""

        self.handoff.zero_()
        if dt_s <= 0.0:
            raise ValueError("takeoff update interval must be positive")
        positions = torch.as_tensor(position, dtype=torch.float32, device=self.device)
        fresh = torch.as_tensor(state_fresh, dtype=torch.bool, device=self.device)
        if positions.shape != (self.num_envs, 3) or fresh.shape != (self.num_envs,):
            raise ValueError("position and freshness must cover every environment")

        active = self.active.clone()
        self.elapsed_time[active] += dt_s
        finite = torch.isfinite(positions).all(dim=-1)
        distance = torch.linalg.vector_norm(positions - self.target_position, dim=-1)
        accepted = active & fresh & finite & (distance <= self.cfg.acceptance_distance_m)
        self.acceptance_time[active] = torch.where(
            accepted[active],
            self.acceptance_time[active] + dt_s,
            torch.zeros_like(self.acceptance_time[active]),
        )

        completed_now = active & (self.acceptance_time >= self.cfg.acceptance_time_s)
        self.complete |= completed_now
        self.handoff |= completed_now
        self.active &= ~completed_now

        if self.cfg.timeout_s is not None:
            timed_out_now = self.active & (self.elapsed_time >= self.cfg.timeout_s)
            self.timed_out |= timed_out_now
            self.active &= ~timed_out_now


class TakeoffStage(IntEnum):
    """Starling pre-mission sequence states."""

    IDLE = 0
    REQUEST_CONTROL = 1
    ARM = 2
    TAKEOFF = 3
    COMPLETE = 4
    FAILED = 5


@dataclass(frozen=True)
class TakeoffExecutorCfg:
    """Sequence configuration with no invented deployment timeout."""

    setup_timeout_s: float | None = None
    takeoff: TakeoffCfg = field(default_factory=TakeoffCfg)

    def __post_init__(self) -> None:
        if self.setup_timeout_s is not None and self.setup_timeout_s <= 0.0:
            raise ValueError("takeoff setup timeout must be positive when configured")


class PremissionTakeoffExecutor:
    """Batched control/arm/takeoff handoff matching Starling's behavior order."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cfg: TakeoffExecutorCfg | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or TakeoffExecutorCfg()
        self.takeoff = PremissionTakeoff(num_envs, self.device, self.cfg.takeoff)
        self.stage = torch.full((num_envs,), int(TakeoffStage.IDLE), dtype=torch.long, device=self.device)
        self.setup_elapsed = torch.zeros(num_envs, device=self.device)
        self.request_control = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.request_arm = torch.zeros_like(self.request_control)
        self.takeoff_started = torch.zeros_like(self.request_control)
        self.handoff = torch.zeros_like(self.request_control)

    @property
    def target_position(self) -> torch.Tensor:
        return self.takeoff.target_position

    @property
    def target_yaw(self) -> torch.Tensor:
        return self.takeoff.target_yaw

    @property
    def target_active(self) -> torch.Tensor:
        return self.stage == int(TakeoffStage.TAKEOFF)

    @property
    def policy_enabled(self) -> torch.Tensor:
        return self.stage == int(TakeoffStage.COMPLETE)

    @property
    def failed(self) -> torch.Tensor:
        return self.stage == int(TakeoffStage.FAILED)

    def start(self, env_ids: torch.Tensor | Sequence[int]) -> None:
        """Start selected environments at the control-acquisition boundary."""

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.stage[ids] = int(TakeoffStage.REQUEST_CONTROL)
        self.setup_elapsed[ids] = 0.0
        self.request_control[ids] = False
        self.request_arm[ids] = False
        self.takeoff_started[ids] = False
        self.handoff[ids] = False

    def update(
        self,
        dt_s: float,
        *,
        position: torch.Tensor,
        yaw: torch.Tensor,
        state_fresh: torch.Tensor,
        has_control: torch.Tensor,
        armed: torch.Tensor,
    ) -> None:
        """Advance confirmed status and emit command masks for one control update."""

        if dt_s <= 0.0:
            raise ValueError("takeoff executor update interval must be positive")
        positions = torch.as_tensor(position, dtype=torch.float32, device=self.device)
        yaws = torch.as_tensor(yaw, dtype=torch.float32, device=self.device)
        fresh = torch.as_tensor(state_fresh, dtype=torch.bool, device=self.device)
        control = torch.as_tensor(has_control, dtype=torch.bool, device=self.device)
        is_armed = torch.as_tensor(armed, dtype=torch.bool, device=self.device)
        if positions.shape != (self.num_envs, 3):
            raise ValueError("position must cover every environment")
        if any(value.shape != (self.num_envs,) for value in (yaws, fresh, control, is_armed)):
            raise ValueError("yaw and status masks must cover every environment")

        finite = torch.isfinite(positions).all(dim=-1) & torch.isfinite(yaws)
        usable_state = fresh & finite
        self.takeoff_started.zero_()
        self.handoff.zero_()

        setup_active = (self.stage == int(TakeoffStage.REQUEST_CONTROL)) | (self.stage == int(TakeoffStage.ARM))
        self.setup_elapsed[setup_active] += dt_s

        control_confirmed = (self.stage == int(TakeoffStage.REQUEST_CONTROL)) & usable_state & control
        self.stage = torch.where(control_confirmed, torch.full_like(self.stage, int(TakeoffStage.ARM)), self.stage)
        ready = (self.stage == int(TakeoffStage.ARM)) & usable_state & control & is_armed
        self.takeoff.start_mask(ready, positions, yaws)
        self.stage = torch.where(ready, torch.full_like(self.stage, int(TakeoffStage.TAKEOFF)), self.stage)
        self.takeoff_started |= ready

        taking_off = self.stage == int(TakeoffStage.TAKEOFF)
        takeoff_state_valid = usable_state & control & is_armed
        self.takeoff.update(dt_s, position=positions, state_fresh=takeoff_state_valid)
        completed = taking_off & self.takeoff.handoff
        timed_out = taking_off & self.takeoff.timed_out
        self.stage = torch.where(completed, torch.full_like(self.stage, int(TakeoffStage.COMPLETE)), self.stage)
        self.stage = torch.where(timed_out, torch.full_like(self.stage, int(TakeoffStage.FAILED)), self.stage)
        self.handoff |= completed

        if self.cfg.setup_timeout_s is not None:
            setup_timed_out = setup_active & (self.setup_elapsed >= self.cfg.setup_timeout_s)
            # Reaching TAKEOFF on this update wins over the setup timer.
            setup_timed_out &= self.stage != int(TakeoffStage.TAKEOFF)
            self.stage = torch.where(setup_timed_out, torch.full_like(self.stage, int(TakeoffStage.FAILED)), self.stage)

        sequenced = (
            (self.stage == int(TakeoffStage.REQUEST_CONTROL))
            | (self.stage == int(TakeoffStage.ARM))
            | (self.stage == int(TakeoffStage.TAKEOFF))
        )
        self.request_control = sequenced & usable_state & ~control
        self.request_arm = (
            ((self.stage == int(TakeoffStage.ARM)) | (self.stage == int(TakeoffStage.TAKEOFF)))
            & usable_state
            & control
            & ~is_armed
        )
