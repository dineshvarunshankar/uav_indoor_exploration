"""Accounting helpers for reward-control evidence."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def split_reward_totals(
    weighted_sums: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    env_ids: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return integrated raw and weighted totals for the selected environments."""
    raw: dict[str, torch.Tensor] = {}
    weighted: dict[str, torch.Tensor] = {}
    for name, values in weighted_sums.items():
        weight = float(weights[name])
        if weight == 0.0:
            raise ValueError(f"cannot recover raw reward for zero-weight term {name!r}")
        selected = values[env_ids].clone()
        weighted[name] = selected
        raw[name] = selected / weight
    return raw, weighted
