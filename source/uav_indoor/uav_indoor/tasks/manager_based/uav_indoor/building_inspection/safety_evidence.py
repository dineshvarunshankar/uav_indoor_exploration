"""Safety verdicts that keep unknown apart from safe."""

from __future__ import annotations

import math
from collections.abc import Sequence

PASS = "PASS"
FAIL = "FAIL"
UNVERIFIED = "UNVERIFIED"

DEFAULT_MIN_OBSERVABILITY = 0.95


def crossing_observability(records: Sequence[dict]) -> float:
    """Fraction of scored steps the crossing test could answer."""

    scored = sum(int(r.get("crossing_scored_steps", 0)) for r in records)
    observable = sum(int(r.get("crossing_observable_steps", 0)) for r in records)
    if scored <= 0:
        return float("nan")
    return observable / scored


def safety_verdict(
    failures: Sequence[str],
    records: Sequence[dict],
    min_observability: float = DEFAULT_MIN_OBSERVABILITY,
) -> tuple[str, list[str]]:
    """PASS, FAIL or UNVERIFIED, with notes explaining an unverified one."""

    if failures:
        return FAIL, []
    if not records:
        return UNVERIFIED, ["no episode records, so nothing about safety was measured"]
    rate = crossing_observability(records)
    if math.isnan(rate):
        return UNVERIFIED, ["no crossing-test step was recorded, so safety was not measured"]
    if rate < min_observability:
        return UNVERIFIED, [
            f"the crossing test could answer {rate:.3f} of steps, under the required "
            f"{min_observability:.3f}: safety is unverified, not clean"
        ]
    return PASS, []
