"""Pure Torch state and behavior for a vectorized crowd."""

from __future__ import annotations

import math
from enum import IntEnum

import torch


class Behavior(IntEnum):
    """Available person behaviors."""

    IDLE = 0
    PATROL = 1
    WANDER = 2
    LOOK_AROUND = 3


class Posture(IntEnum):
    """Casualty posture, as a tilt away from upright.

    The character asset is a standing rig, so a posture is a rotation of that rig. That
    gives the right geometry -- body centre height and apparent extent, which is what the
    camera and the height servo see -- but not a natural pose.
    """

    STANDING = 0
    SITTING = 1
    LYING = 2


POSTURE_TILT_RAD = (0.0, math.radians(55.0), math.radians(90.0))
PERSON_HALF_HEIGHT_M = 0.85
PERSON_RADIUS_M = 0.30


class CrowdManager:
    """Own bounded motion state for ten people in each environment."""

    num_people = 10
    # Every person is a casualty. The end-to-end mission inspects all ten, so there are no
    # bystanders; the moving vehicle and rogue drone supply the dynamic obstacles.
    num_targets = 10

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        env_origins: torch.Tensor | None = None,
        seed: int = 0,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)

        if env_origins is None:
            env_origins = torch.zeros(num_envs, 3, device=self.device)
        if env_origins.shape != (num_envs, 3):
            raise ValueError(f"env_origins must have shape ({num_envs}, 3)")
        self.env_origins = env_origins.to(self.device).clone()

        # Must match PERSON_POSITIONS in the scene config: the animation runtime owns the
        # real positions and overwrites this ring after the first step.
        bearings = [0, 41, 77, 108, 145, 183, 219, 256, 291, 325]
        angle = torch.deg2rad(torch.tensor(bearings, device=self.device, dtype=torch.float32))
        radius = torch.tensor([16.0, 22.0, 27.0, 14.0, 25.0, 19.0, 28.0, 15.0, 23.0, 20.0], device=self.device)
        local = torch.zeros(self.num_people, 3, device=self.device)
        local[:, 0] = radius * torch.cos(angle)
        local[:, 1] = radius * torch.sin(angle)
        self.position = self.env_origins[:, None, :] + local[None, :, :]
        self.position[..., 2] = self.env_origins[:, None, 2]
        self.anchor = self.position.clone()

        self.is_target = torch.zeros(num_envs, self.num_people, dtype=torch.bool, device=self.device)
        self.is_target[:, : self.num_targets] = True
        initial_behavior = torch.tensor(
            [
                # Casualties do not walk, so the GPS seed stays accurate.
                *([Behavior.IDLE] * 10),
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._advanced_behavior = initial_behavior.expand(num_envs, -1).clone()
        self.behavior = self._advanced_behavior.clone()
        self._last_curriculum_stage = -1
        self.speed = 0.65 + 0.55 * torch.rand(
            num_envs,
            self.num_people,
            generator=self._generator,
            device=self.device,
        )
        # Facing and posture vary per person per environment. Without this the drone sees
        # one casualty pose from one angle and learns only that.
        self.yaw = 2.0 * math.pi * torch.rand(
            num_envs, self.num_people, generator=self._generator, device=self.device
        )
        self.posture = torch.zeros(num_envs, self.num_people, dtype=torch.long, device=self.device)
        casualty_posture = torch.randint(
            0, len(POSTURE_TILT_RAD), (num_envs, self.num_targets),
            generator=self._generator, device=self.device,
        )
        self.posture[:, : self.num_targets] = casualty_posture
        self.destination = self.position.clone()
        self._select_destination(torch.ones_like(self.is_target))

    def _select_destination(self, mask: torch.Tensor) -> None:
        """Select a local destination without changing a person's position."""

        angle = 2.0 * math.pi * torch.rand(
            self.num_envs,
            self.num_people,
            generator=self._generator,
            device=self.device,
        )
        radius = 5.0 + 8.0 * torch.rand(
            self.num_envs,
            self.num_people,
            generator=self._generator,
            device=self.device,
        )
        candidate = self.env_origins[:, None, :].expand(-1, self.num_people, -1).clone()
        candidate[..., 0] += radius * torch.cos(angle)
        candidate[..., 1] += radius * torch.sin(angle)
        target_mask = mask & self.is_target
        patrol_angle = 2.0 * math.pi * torch.rand(
            self.num_envs,
            self.num_people,
            generator=self._generator,
            device=self.device,
        )
        patrol_radius = 1.5 * torch.rand(
            self.num_envs,
            self.num_people,
            generator=self._generator,
            device=self.device,
        )
        patrol_candidate = self.anchor.clone()
        patrol_candidate[..., 0] += patrol_radius * torch.cos(patrol_angle)
        patrol_candidate[..., 1] += patrol_radius * torch.sin(patrol_angle)
        candidate[target_mask] = patrol_candidate[target_mask]
        self.destination[mask] = candidate[mask]

    def step(self, dt: float, curriculum_stage: int) -> None:
        """Advance all people with a speed-bounded position update."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if curriculum_stage == 0:
            self.behavior[:] = int(Behavior.IDLE)
        elif self._last_curriculum_stage == 0:
            self.behavior.copy_(self._advanced_behavior)
        self._last_curriculum_stage = curriculum_stage

        offset = self.destination - self.position
        distance = torch.linalg.vector_norm(offset[..., :2], dim=-1)
        moving = (self.behavior == int(Behavior.PATROL)) | (self.behavior == int(Behavior.WANDER))
        moving &= distance > 0.05
        direction = offset / distance.clamp_min(1.0e-6)[..., None]
        travel = torch.minimum(self.speed * dt, distance)
        self.position += direction * (travel * moving)[..., None]
        self.position[..., 2] = self.env_origins[:, None, 2]
        self.yaw = torch.where(moving, torch.atan2(direction[..., 1], direction[..., 0]), self.yaw)

        reached = moving & (distance <= self.speed * dt + 0.05)
        if reached.any():
            self._select_destination(reached)

    def on_drone_reset(self, env_ids: torch.Tensor) -> None:
        """Keep crowd state continuous when aircraft episodes reset."""

        del env_ids

    @property
    def tilt(self) -> torch.Tensor:
        """Tilt away from upright, in radians, for each person."""

        table = torch.tensor(POSTURE_TILT_RAD, device=self.device)
        return table[self.posture]

    @property
    def ground_clearance_m(self) -> torch.Tensor:
        """Lift needed so a tilted body rests on the ground instead of sinking into it.

        The character pivot is at the feet. Rotating a standing rig about that pivot puts
        the body axis in the ground plane, so half the body ends up below z = 0.
        """

        return PERSON_RADIUS_M * torch.sin(self.tilt)

    @property
    def body_centre_height_m(self) -> torch.Tensor:
        """Height of the body centre above ground, given the posture.

        Upright the centre sits at half the body height. Flat it sits at the body radius.
        """

        return PERSON_RADIUS_M + (PERSON_HALF_HEIGHT_M - PERSON_RADIUS_M) * torch.cos(self.tilt)

    def orientation_wxyz(self) -> torch.Tensor:
        """Return yaw rotation followed by the posture tilt, as ``R_z(yaw) * R_x(tilt)``."""

        half_yaw, half_tilt = 0.5 * self.yaw, 0.5 * self.tilt
        cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)
        ct, st = torch.cos(half_tilt), torch.sin(half_tilt)
        return torch.stack((cy * ct, cy * st, sy * st, sy * ct), dim=-1)
