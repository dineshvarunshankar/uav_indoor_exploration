#!/usr/bin/env python3
"""Run one detector-shaped opening through validation, observation, and safe exit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import NamedTuple

import torch

ROOT = Path(__file__).resolve().parents[1]
INSPECTION_DIR = (
    ROOT / "source/uav_indoor/uav_indoor/tasks/manager_based/uav_indoor/inspection"
)
DT = 0.5


def _load_inspection():
    name = "building_inspection_stage3_core"
    spec = importlib.util.spec_from_file_location(
        name,
        INSPECTION_DIR / "__init__.py",
        submodule_search_locations=[str(INSPECTION_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


inspection = _load_inspection()


class SmokeResult(NamedTuple):
    phases: tuple[str, ...]
    observe_s: float
    validated: bool
    inspected: bool
    exit_complete: bool
    recovery_failed: bool


def run_smoke(max_steps: int = 1200) -> SmokeResult:
    """Follow mission references exactly and return the resulting stage-3 evidence."""

    MissionPhase = inspection.MissionPhase
    MissionStage = inspection.MissionStage
    OpeningStatus = inspection.OpeningStatus
    supervisor = inspection.BuildingMissionSupervisor(
        1,
        "cpu",
        inspection.BuildingMissionCfg(initial_stage=int(MissionStage.FULL)),
    )
    supervisor.reset(survey_direction=torch.ones(1))

    position = torch.tensor([[-50.0, 0.0, 5.0]])
    yaw = torch.zeros(1)
    standoff = torch.tensor([[0.0, 0.0, 5.0]])
    opening = torch.tensor([[5.0, 0.0, 5.0]])
    facade_normal = torch.tensor([[-1.0, 0.0, 0.0]])
    phases = [MissionPhase(int(supervisor.phase[0])).name]
    observe_s = 0.0
    time_s = 0.0

    for _ in range(max_steps):
        before = MissionPhase(int(supervisor.phase[0]))
        ray = opening - position
        ray = (ray / ray.norm(dim=-1, keepdim=True))[:, None, :]
        reference = supervisor.update(
            DT,
            robot_position_w=position,
            robot_yaw_w=yaw,
            building_standoff_w=standoff,
            opening_origin_w=position,
            opening_ray_w=ray,
            # Live detector track and box.
            opening_track_id=torch.zeros(1, 1, dtype=torch.long),
            opening_box_xywhn=torch.tensor([[[0.5, 0.5, 0.25, 0.30]]]),
            opening_angular_wh=torch.full((1, 1, 2), 0.2),
            opening_confidence=torch.full((1, 1), 0.9),
            opening_valid=torch.ones(1, 1, dtype=torch.bool),
            opening_box_centered=torch.ones(1, dtype=torch.bool),
            person_track_id=torch.full((1, 1), -1, dtype=torch.long),
            person_box_xywhn=torch.zeros(1, 1, 4),
            person_confidence=torch.zeros(1, 1),
            person_age_s=torch.zeros(1, 1),
            person_valid=torch.zeros(1, 1, dtype=torch.bool),
            tof_clear=torch.ones(1, dtype=torch.bool),
            tof_valid=torch.ones(1, dtype=torch.bool),
            clearance_m=torch.full((1,), 3.0),
            clearance_known=torch.ones(1, dtype=torch.bool),
            avoidance_active=torch.zeros(1, dtype=torch.bool),
            time_s=time_s,
            facade_normal_w=facade_normal,
            facade_offset_m=torch.tensor([-5.0]),
            facade_valid=torch.ones(1, dtype=torch.bool),
        )
        if before is MissionPhase.OBSERVE:
            observe_s += DT
        after = MissionPhase(int(supervisor.phase[0]))
        if after.name != phases[-1]:
            phases.append(after.name)
        position = reference.position_w.clone()
        yaw = reference.yaw_w.clone()
        time_s += DT
        if after is MissionPhase.RESUME_SURVEY:
            break
    else:
        raise RuntimeError(
            f"stage-3 smoke exceeded {max_steps} steps in "
            f"{MissionPhase(int(supervisor.phase[0])).name}"
        )

    slot = int(supervisor.active_slot[0])
    return SmokeResult(
        phases=tuple(phases),
        observe_s=observe_s,
        validated=bool(supervisor.ledger.validated[0, slot]),
        inspected=int(supervisor.ledger.status[0, slot]) == int(OpeningStatus.INSPECTED),
        exit_complete=bool(supervisor.breadcrumbs.exit_complete[0]),
        recovery_failed=bool(supervisor.recovery_failed[0]),
    )


def main() -> None:
    result = run_smoke()
    print(
        "PASS: stage-3 mission sequence "
        f"{' -> '.join(result.phases)}; observe={result.observe_s:.1f}s; "
        f"validated={result.validated}; inspected={result.inspected}; "
        f"exit_complete={result.exit_complete}; recovery_failed={result.recovery_failed}"
    )


if __name__ == "__main__":
    main()
