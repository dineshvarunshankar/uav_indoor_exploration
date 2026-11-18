"""Contact-force reduction across one complete policy step."""

from __future__ import annotations

import torch


def control_step_maximum_contact_force(
    net_forces_w_history: torch.Tensor,
    physics_steps_per_control_step: int,
) -> torch.Tensor:
    """Return each environment's maximum contact force during one control step."""

    if net_forces_w_history.ndim != 4 or net_forces_w_history.shape[-1] != 3:
        raise ValueError("contact-force history must have shape (N, H, B, 3)")
    if physics_steps_per_control_step <= 0:
        raise ValueError("physics steps per control step must be positive")
    if net_forces_w_history.shape[1] < physics_steps_per_control_step:
        raise ValueError(
            "contact-force history must retain at least "
            f"{physics_steps_per_control_step} physics samples"
        )
    forces = net_forces_w_history[:, :physics_steps_per_control_step]
    return torch.linalg.vector_norm(forces, dim=-1).amax(dim=(1, 2))
