"""Rough opening proposals shared by temporary YAML and future reconstruction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import torch
import yaml


class OpeningProposals(NamedTuple):
    """Identifiers and world-frame rough positions; no aperture geometry."""

    ids: tuple[str, ...]
    position_w: torch.Tensor


class OpeningProposalSource(Protocol):
    """Boundary a future offline reconstruction adapter must implement."""

    def load(self) -> OpeningProposals: ...


def parse_opening_proposals(document: Mapping[str, Any]) -> OpeningProposals:
    """Validate the shared proposal document and return immutable CPU data."""
    if set(document) != {"openings"}:
        raise ValueError("proposal document must contain only openings")
    records = document["openings"]
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("openings must be a sequence")

    ids: list[str] = []
    positions: list[list[float]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {"id", "position"}:
            raise ValueError(f"opening {index} must contain only id and position")
        opening_id = record["id"]
        if not isinstance(opening_id, str) or not opening_id.strip():
            raise ValueError(f"opening {index} id must be a non-empty string")
        position = record["position"]
        if not isinstance(position, Sequence) or isinstance(position, (str, bytes)) or len(position) != 3:
            raise ValueError(f"opening {opening_id!r} position must contain three coordinates")
        try:
            coordinates = [float(value) for value in position]
        except (TypeError, ValueError) as error:
            raise ValueError(f"opening {opening_id!r} position must be numeric") from error
        ids.append(opening_id)
        positions.append(coordinates)

    if len(set(ids)) != len(ids):
        raise ValueError("duplicate opening id")
    position_w = torch.tensor(positions, dtype=torch.float32).reshape(-1, 3)
    if not torch.isfinite(position_w).all():
        raise ValueError("opening positions must be finite")
    return OpeningProposals(tuple(ids), position_w)


def load_opening_proposals(path: str | Path) -> OpeningProposals:
    """Load temporary Xform positions through the reconstruction-compatible boundary."""
    with Path(path).open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, Mapping):
        raise ValueError("proposal document must be a mapping")
    return parse_opening_proposals(document)


def to_environment_local(position_w: torch.Tensor, env_origins_w: torch.Tensor) -> torch.Tensor:
    """Convert shared world positions to every simulator environment local frame."""
    if position_w.ndim != 2 or position_w.shape[-1] != 3:
        raise ValueError("position_w must have shape (openings, 3)")
    if env_origins_w.ndim != 2 or env_origins_w.shape[-1] != 3:
        raise ValueError("env_origins_w must have shape (environments, 3)")
    return position_w.to(env_origins_w.device)[None, :, :] - env_origins_w[:, None, :]


def perturb_training_positions(
    position_local: torch.Tensor,
    training: bool,
    max_abs_error_m: tuple[float, float, float] | None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply bounded uniform proposal error only when a measured bound is configured."""
    if not training or max_abs_error_m is None:
        return position_local.clone()
    bound = torch.as_tensor(max_abs_error_m, dtype=position_local.dtype, device=position_local.device)
    if bound.shape != (3,) or not torch.isfinite(bound).all() or (bound < 0.0).any():
        raise ValueError("max_abs_error_m must contain three finite non-negative bounds")
    sample = torch.rand(
        position_local.shape,
        dtype=position_local.dtype,
        device=position_local.device,
        generator=generator,
    )
    return position_local + (2.0 * sample - 1.0) * bound


def validate_association_residual(value: float | None) -> float | None:
    """Validate the measured distance limit used for automatic hint matching."""

    if value is None:
        return None
    residual = float(value)
    if not torch.isfinite(torch.tensor(residual)) or residual <= 0.0:
        raise ValueError("association residual must be finite and positive")
    return residual
