"""Strict JSON serialization for gate artifacts."""

from __future__ import annotations

import json
import math
from typing import Any


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with explicit missing values."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def dumps_strict(payload: Any, **kwargs: Any) -> str:
    """Return standards-compliant JSON without NaN or Infinity tokens."""

    return json.dumps(_json_safe(payload), allow_nan=False, **kwargs)
