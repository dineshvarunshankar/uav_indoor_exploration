"""Simulator truth used only by reset, critic observations, and scoring."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

_TRUTH_FILE = Path(__file__).resolve().parents[1] / "west_face_openings.yaml"


def load_opening_truth(path: Path = _TRUTH_FILE) -> torch.Tensor:
    """Load the measured construction-site openings as an immutable CPU tensor."""

    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)["openings"]
    return torch.tensor(values, dtype=torch.float32)


OPENINGS = load_opening_truth()
OPENING_TRUTH_COUNT = int(OPENINGS.shape[0])


def ledger_truth_matches(
    ledger_position_w: torch.Tensor,
    eligible: torch.Tensor,
    truth_position_w: torch.Tensor,
    truth_count: torch.Tensor,
    radius_m: float,
) -> torch.Tensor:
    """Match ledger entries to privileged simulator truth."""

    if truth_position_w.shape[1] == 0:
        return torch.zeros_like(eligible)
    distance = torch.linalg.vector_norm(ledger_position_w[:, :, None, :] - truth_position_w[:, None, :, :], dim=-1)
    truth_index = torch.arange(truth_position_w.shape[1], device=truth_position_w.device)
    known = truth_index[None, :] < truth_count[:, None]
    nearest = distance.masked_fill(~known[:, None, :], torch.inf).amin(dim=2)
    return eligible & (nearest <= radius_m)
