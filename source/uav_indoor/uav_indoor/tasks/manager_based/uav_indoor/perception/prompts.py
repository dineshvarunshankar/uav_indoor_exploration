"""Static YOLOE prompt vocabulary and mission roles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class DetectionRole(IntEnum):
    """Mission role assigned to one prompted detection class.

    Integer values index device tensors, so the order is part of the interface.
    """

    OPENING_CANDIDATE = 0
    PERSON = 1


@dataclass(frozen=True)
class PromptSpec:
    """One text prompt and its mission role."""

    text: str
    role: DetectionRole


@dataclass(frozen=True)
class PromptVocabulary:
    """Ordered prompt definitions that match YOLOE class indices."""

    specifications: tuple[PromptSpec, ...]

    @classmethod
    def default(cls) -> PromptVocabulary:
        """Return the initial zero-shot vocabulary for the building task."""
        opening = DetectionRole.OPENING_CANDIDATE
        return cls(
            (
                PromptSpec("open window", opening),
                PromptSpec("open doorway", opening),
                PromptSpec("broken window", opening),
                PromptSpec("building entrance", opening),
                PromptSpec("glass window", opening),
                PromptSpec("person", DetectionRole.PERSON),
            )
        )

    @property
    def prompts(self) -> tuple[str, ...]:
        """Return prompt text in model class-index order."""
        return tuple(specification.text for specification in self.specifications)

    def role_for_class_id(self, class_id: int) -> DetectionRole:
        """Return the mission role for a YOLOE class index."""
        return self.specifications[class_id].role

    def role_lookup(self, device: torch.device | str) -> torch.Tensor:
        """Return a ``(num_classes,)`` long tensor that maps a class index to a role."""

        return torch.tensor(
            [int(specification.role) for specification in self.specifications],
            dtype=torch.long,
            device=device,
        )
