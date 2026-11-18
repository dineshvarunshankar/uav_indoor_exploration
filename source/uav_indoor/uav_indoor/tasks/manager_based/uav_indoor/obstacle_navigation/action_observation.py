"""Select the action that produced the actor's current vehicle state."""

from __future__ import annotations

import torch


def latest_policy_action(action_manager, term_name: str) -> torch.Tensor:
    """Return the most recently processed command for one action term."""

    term = action_manager.get_term(term_name)
    return term.raw_actions.clamp(-1.0, 1.0)
