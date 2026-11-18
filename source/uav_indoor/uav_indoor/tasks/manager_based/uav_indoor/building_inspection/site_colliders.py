"""Author collision geometry on the construction site, in the local stage only."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

DEFAULT_INCLUDE: tuple[str, ...] = ("*",)
TRIANGLE_MESH = "none"


@dataclass
class SiteColliderCfg:
    enabled: bool = True
    approximation: str = TRIANGLE_MESH
    include: tuple[str, ...] = DEFAULT_INCLUDE
    exclude: tuple[str, ...] = ()
    deinstance: tuple[str, ...] = ("*",)


@dataclass
class SiteColliderReport:
    approximation: str = TRIANGLE_MESH
    root: str = ""
    applied: list[str] = field(default_factory=list)
    skipped_not_included: int = 0
    skipped_excluded: list[str] = field(default_factory=list)
    skipped_instance_proxy: list[str] = field(default_factory=list)
    deinstanced: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "approximation": self.approximation,
            "collision_geometry_source": "rendered_usd_mesh_prim",
            "applied": len(self.applied),
            "skipped_not_included": self.skipped_not_included,
            "skipped_excluded": len(self.skipped_excluded),
            "skipped_instance_proxy": len(self.skipped_instance_proxy),
            "deinstanced": len(self.deinstanced),
            "by_group": self.by_group(),
        }

    def by_group(self) -> dict[str, dict[str, int]]:
        groups: dict[str, dict[str, int]] = {}
        for kind, paths in (
            ("applied", self.applied),
            ("instance_proxy", self.skipped_instance_proxy),
            ("excluded", self.skipped_excluded),
        ):
            for path in paths:
                entry = groups.setdefault(group_of(path, self.root), {})
                entry[kind] = entry.get(kind, 0) + 1
        return dict(sorted(groups.items()))


def group_of(prim_path: str, root: str) -> str:
    """The first path segment below ``root``."""

    if not root or not prim_path.startswith(root):
        return next((p for p in prim_path.split("/") if p), prim_path)
    parts = [p for p in prim_path[len(root):].split("/") if p]
    if parts and parts[0] == root.rsplit("/", 1)[-1]:
        parts = parts[1:]
    return parts[0] if parts else prim_path


def selects(prim_path: str, include: Sequence[str], exclude: Sequence[str]) -> bool:
    if any(fnmatch.fnmatch(prim_path, pattern) for pattern in exclude):
        return False
    return any(fnmatch.fnmatch(prim_path, pattern) for pattern in include)


def plan_colliders(
    mesh_paths: Sequence[str],
    cfg: SiteColliderCfg,
    is_instance_proxy: Callable[[str], bool] = lambda _: False,
    root: str = "",
) -> SiteColliderReport:
    """Decide which meshes get a collider."""

    report = SiteColliderReport(approximation=cfg.approximation, root=root)
    for path in mesh_paths:
        if any(fnmatch.fnmatch(path, pattern) for pattern in cfg.exclude):
            report.skipped_excluded.append(path)
        elif not any(fnmatch.fnmatch(path, pattern) for pattern in cfg.include):
            report.skipped_not_included += 1
        elif is_instance_proxy(path):
            report.skipped_instance_proxy.append(path)
        else:
            report.applied.append(path)
    return report


def open_run_through_centre(free_offsets: Sequence[float], step: float) -> list[float]:
    """The contiguous run of free offsets containing 0, from a scan across an opening."""

    run: list[float] = []
    for offset in sorted(free_offsets):
        if run and abs(offset - run[-1]) > step * 1.5:
            if run[0] <= 0.0 <= run[-1]:
                return run
            run = []
        run.append(offset)
    return run if run and run[0] <= 0.0 <= run[-1] else []


def strongest_contact_offset(samples: Sequence[tuple[float, float]]) -> float | None:
    """Return the offset with the largest measured contact force."""

    return max(samples, key=lambda sample: sample[1])[0] if samples else None


def apply_site_colliders(stage, root_path: str, cfg: SiteColliderCfg) -> SiteColliderReport:
    """Author colliders on the live stage."""

    from omni.physx.scripts import utils as physx_utils
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    report = SiteColliderReport(approximation=cfg.approximation, root=root_path)

    seen: set[str] = set()
    while True:
        instance_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.IsInstance()
            and any(fnmatch.fnmatch(str(prim.GetPath()), pattern) for pattern in cfg.deinstance)
            and str(prim.GetPath()) not in seen
        ]
        if not instance_paths:
            break
        changed = False
        for path in instance_paths:
            seen.add(path)
            prim = stage.GetPrimAtPath(path)
            if prim and prim.SetInstanceable(False):
                report.deinstanced.append(path)
                changed = True
        if not changed:
            break

    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for prim in Usd.PrimRange(root, predicate):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        if any(fnmatch.fnmatch(path, pattern) for pattern in cfg.exclude):
            report.skipped_excluded.append(path)
        elif not any(fnmatch.fnmatch(path, pattern) for pattern in cfg.include):
            report.skipped_not_included += 1
        elif prim.IsInstanceProxy():
            report.skipped_instance_proxy.append(path)
        else:
            physx_utils.setCollider(prim, approximationShape=cfg.approximation)
            report.applied.append(path)
    return report


LAST_REPORT: dict[str, SiteColliderReport] = {}


def spawn_site(prim_path, spawn_cfg, translation=None, orientation=None, **kwargs):
    """Spawn the site and author its configured colliders before physics starts."""

    import isaacsim.core.utils.stage as stage_utils

    import isaaclab.sim as sim_utils

    cfg = getattr(spawn_cfg, "site_colliders", None)
    if not isinstance(cfg, SiteColliderCfg):
        raise TypeError("site spawn configuration must carry SiteColliderCfg")
    prim = sim_utils.spawn_from_usd(prim_path, spawn_cfg, translation, orientation, **kwargs)
    LAST_REPORT[prim_path] = apply_site_colliders(
        stage_utils.get_current_stage(), prim_path, cfg
    )
    return prim


def enable_on(env_cfg) -> None:
    """Configure one module-resolvable site spawner before the environment is built."""

    env_cfg.scene.scene.spawn.site_colliders = env_cfg.site_colliders
    env_cfg.scene.scene.spawn.func = spawn_site
    env_cfg.scene.scene.collision_group = -1
