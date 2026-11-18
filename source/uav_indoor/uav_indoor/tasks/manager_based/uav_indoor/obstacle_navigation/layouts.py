"""Seeded, clearance-aware layouts for random-goal obstacle navigation."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Generic, TypeVar

import numpy as np

_LayoutT = TypeVar("_LayoutT")


class LazyLayoutBank(Generic[_LayoutT]):
    """Build and cache one deterministic layout only when it is selected."""

    def __init__(self, size: int, factory: Callable[[int], _LayoutT]) -> None:
        if size <= 0:
            raise ValueError("layout bank size must be positive")
        self._factory = factory
        self._items: list[_LayoutT | None] = [None] * size

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> _LayoutT:
        if not isinstance(index, int):
            raise TypeError("layout bank index must be an integer")
        if index < 0:
            index += len(self._items)
        if index < 0 or index >= len(self._items):
            raise IndexError(index)
        item = self._items[index]
        if item is None:
            item = self._factory(index)
            self._items[index] = item
        return item


class LayoutDifficulty(IntEnum):
    OPEN = 0
    SPARSE = 1
    MEDIUM = 2
    DENSE = 3


@dataclass(frozen=True)
class ObstacleSpec:
    """One visible primitive and its matching collider dimensions."""

    family: str
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    slot_index: int = -1


@dataclass(frozen=True)
class NavigationLayout:
    """One accepted scene and its hidden feasibility reference."""

    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    start_m: tuple[float, float, float]
    goal_m: tuple[float, float, float]
    obstacles: tuple[ObstacleSpec, ...]
    reference_path: tuple[tuple[float, float, float], ...]
    route_family: str = "unclassified"
    detour_direction: str = "unclassified"
    narrow_passage: bool = False
    direct_path_blocked: bool = False
    minimum_route_clearance_m: float = math.inf
    route_evidence: str = "unmeasured"
    detour_side: str = "unclassified"

    def point_is_inside_flight_volume(
        self, point_m: tuple[float, float, float], clearance_m: float
    ) -> bool:
        """Return whether the vehicle envelope stays inside containment and above 2.5 m."""

        point = np.asarray(point_m, dtype=np.float64)
        low = np.asarray(self.bounds_min_m, dtype=np.float64) + clearance_m
        low[2] = max(low[2], _MINIMUM_FLIGHT_HEIGHT_M)
        high = np.asarray(self.bounds_max_m, dtype=np.float64) - clearance_m
        return bool(np.all(point >= low) and np.all(point <= high))

    def point_is_clear(self, point_m: tuple[float, float, float], clearance_m: float) -> bool:
        point = np.asarray(point_m, dtype=np.float64)
        return all(_point_obstacle_distance(point, obstacle) >= clearance_m for obstacle in self.obstacles)

    def segment_minimum_clearance(
        self,
        start_m: tuple[float, float, float],
        end_m: tuple[float, float, float],
        sample_spacing_m: float,
    ) -> float:
        """Return sampled physical clearance along a segment, including both endpoints."""

        if sample_spacing_m <= 0.0:
            raise ValueError("sample_spacing_m must be positive")
        start = np.asarray(start_m, dtype=np.float64)
        end = np.asarray(end_m, dtype=np.float64)
        samples = max(1, int(math.ceil(float(np.linalg.norm(end - start)) / sample_spacing_m)))
        minimum = math.inf
        for fraction in np.linspace(0.0, 1.0, samples + 1):
            point = start + fraction * (end - start)
            for obstacle in self.obstacles:
                minimum = min(minimum, _point_obstacle_distance(point, obstacle))
        return minimum

    def path_is_clear(
        self,
        path: tuple[tuple[float, float, float], ...],
        clearance_m: float,
        sample_spacing_m: float,
    ) -> bool:
        """Check the complete sampled route against physical shapes and containment."""

        if not path:
            return False
        for first, second in zip(path, path[1:]):
            length = float(np.linalg.norm(np.asarray(second) - np.asarray(first)))
            samples = max(1, int(math.ceil(length / sample_spacing_m)))
            for fraction in np.linspace(0.0, 1.0, samples + 1):
                point = tuple((np.asarray(first) + fraction * (np.asarray(second) - np.asarray(first))).tolist())
                if not self.point_is_inside_flight_volume(point, clearance_m):
                    return False
            if self.segment_minimum_clearance(first, second, sample_spacing_m) < clearance_m:
                return False
        return self.point_is_inside_flight_volume(path[0], clearance_m)


_BOUNDS_MIN = np.asarray((-10.0, -8.0, 0.0), dtype=np.float64)
_BOUNDS_MAX = np.asarray((10.0, 8.0, 8.0), dtype=np.float64)
_MINIMUM_FLIGHT_HEIGHT_M = 2.5
_GRID_M = 0.5
_OBSTACLE_COUNTS = {
    LayoutDifficulty.OPEN: 0,
    LayoutDifficulty.SPARSE: 6,
    LayoutDifficulty.MEDIUM: 12,
    LayoutDifficulty.DENSE: 20,
}
_SLOT_GEOMETRY = (
    ("box", (1.2, 1.2, 1.2)),
    ("wall", (0.4, 3.5, 3.5)),
    ("post", (0.5, 0.5, 4.0)),
    ("overhead_slab", (3.0, 2.0, 0.4)),
) * 5
_FIELD_LATERAL_HALF_WIDTH_M = 3.0
_FIELD_VERTICAL_HALF_HEIGHT_M = 2.0
_FIELD_ROUTE_END_BUFFER = 0.10


def _point_obstacle_distance(point: np.ndarray, obstacle: ObstacleSpec) -> float:
    """Distance from a point to the physical solid used by the simulator."""

    centre = np.asarray(obstacle.center_m, dtype=np.float64)
    size = np.asarray(obstacle.size_m, dtype=np.float64)
    relative = point - centre
    if obstacle.family == "post":
        radial_outside = max(float(np.linalg.norm(relative[:2])) - size[0] * 0.5, 0.0)
        vertical_outside = max(abs(float(relative[2])) - size[2] * 0.5, 0.0)
        return math.hypot(radial_outside, vertical_outside)
    outside = np.maximum(np.abs(relative) - size * 0.5, 0.0)
    return float(np.linalg.norm(outside))


def layout_bank_index(environment_id: int, episode_count: int, bank_size: int) -> int:
    """Rotate each environment through the complete deterministic layout bank."""

    if environment_id < 0 or episode_count < 0 or bank_size <= 0:
        raise ValueError("environment_id and episode_count must be non-negative; bank_size must be positive")
    return (environment_id + episode_count) % bank_size


def obstacle_slot_specs() -> tuple[ObstacleSpec, ...]:
    """Return the fixed visible/collider geometry available in every environment."""

    return tuple(
        ObstacleSpec(
            family=family,
            center_m=(0.0, 0.0, -20.0),
            size_m=size,
            slot_index=index,
        )
        for index, (family, size) in enumerate(_SLOT_GEOMETRY)
    )


def _selected_slot_indices(
    rng: np.random.Generator,
    count: int,
) -> tuple[int, ...]:
    """Select distinct fixed collider slots without a privileged blocker type."""

    return tuple(int(index) for index in rng.permutation(len(_SLOT_GEOMETRY))[:count])


def _sample_obstacles(
    rng: np.random.Generator,
    count: int,
    start: np.ndarray,
    goal: np.ndarray,
    clearance_m: float,
    slot_indices: tuple[int, ...],
) -> tuple[ObstacleSpec, ...]:
    """Scatter a mixed 3D field around the route."""

    obstacles: list[ObstacleSpec] = []
    route = goal - start
    horizontal = route[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm > 1.0e-6:
        lateral_axis = np.asarray((-horizontal[1], horizontal[0], 0.0)) / horizontal_norm
    else:
        lateral_axis = np.asarray((1.0, 0.0, 0.0))
    direction = route / max(float(np.linalg.norm(route)), 1.0e-6)
    vertical_axis = np.asarray((0.0, 0.0, 1.0))
    vertical_axis -= float(np.dot(vertical_axis, direction)) * direction
    vertical_norm = float(np.linalg.norm(vertical_axis))
    if vertical_norm > 1.0e-6:
        vertical_axis /= vertical_norm
    else:
        vertical_axis = np.asarray((0.0, 1.0, 0.0))

    attempts = 0
    while len(obstacles) < count and attempts < 5000:
        attempts += 1
        slot_index = slot_indices[len(obstacles)]
        family, fixed_size = _SLOT_GEOMETRY[slot_index]
        size = np.asarray(fixed_size)
        low = _BOUNDS_MIN + size * 0.5
        high = _BOUNDS_MAX - size * 0.5
        fraction = rng.uniform(
            _FIELD_ROUTE_END_BUFFER, 1.0 - _FIELD_ROUTE_END_BUFFER
        )
        lateral_offset = rng.uniform(
            -_FIELD_LATERAL_HALF_WIDTH_M, _FIELD_LATERAL_HALF_WIDTH_M
        )
        vertical_offset = rng.uniform(
            -_FIELD_VERTICAL_HALF_HEIGHT_M, _FIELD_VERTICAL_HALF_HEIGHT_M
        )
        centre = (
            start
            + fraction * route
            + lateral_offset * lateral_axis
            + vertical_offset * vertical_axis
        )
        centre = np.clip(centre, low, high)
        candidate = ObstacleSpec(
            family,
            tuple(centre.tolist()),
            tuple(size.tolist()),
            slot_index=slot_index,
        )
        provisional = NavigationLayout(
            tuple(_BOUNDS_MIN),
            tuple(_BOUNDS_MAX),
            tuple(start),
            tuple(goal),
            tuple(obstacles + [candidate]),
            (),
        )
        if provisional.point_is_clear(
            tuple(start), clearance_m
        ) and provisional.point_is_clear(tuple(goal), clearance_m):
            obstacles.append(candidate)
    if len(obstacles) != count:
        raise RuntimeError(f"could not place {count} obstacles after {attempts} attempts")
    return tuple(obstacles)

def _grid_point(index: tuple[int, int, int]) -> np.ndarray:
    return _BOUNDS_MIN + np.asarray(index, dtype=np.float64) * _GRID_M


def _grid_index(point: np.ndarray) -> tuple[int, int, int]:
    index = np.rint((point - _BOUNDS_MIN) / _GRID_M).astype(int)
    return tuple(int(value) for value in index)


def _reference_path(
    layout: NavigationLayout, clearance_m: float
) -> tuple[tuple[float, float, float], ...]:
    shape = tuple((np.rint((_BOUNDS_MAX - _BOUNDS_MIN) / _GRID_M).astype(int) + 1).tolist())
    occupied = np.zeros(shape, dtype=np.bool_)
    axes = [
        _BOUNDS_MIN[axis] + np.arange(shape[axis], dtype=np.float64) * _GRID_M
        for axis in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    low = np.asarray(layout.bounds_min_m, dtype=np.float64) + clearance_m
    low[2] = max(low[2], _MINIMUM_FLIGHT_HEIGHT_M)
    high = np.asarray(layout.bounds_max_m, dtype=np.float64) - clearance_m
    occupied |= np.any((grid < low) | (grid > high), axis=-1)
    for obstacle in layout.obstacles:
        centre = np.asarray(obstacle.center_m, dtype=np.float64)
        size = np.asarray(obstacle.size_m, dtype=np.float64)
        relative = grid - centre
        if obstacle.family == "post":
            radial_outside = np.maximum(np.linalg.norm(relative[..., :2], axis=-1) - size[0] * 0.5, 0.0)
            vertical_outside = np.maximum(np.abs(relative[..., 2]) - size[2] * 0.5, 0.0)
            distance = np.hypot(radial_outside, vertical_outside)
        else:
            outside = np.maximum(np.abs(relative) - size * 0.5, 0.0)
            distance = np.linalg.norm(outside, axis=-1)
        occupied |= distance < clearance_m

    start = _grid_index(np.asarray(layout.start_m))
    goal = _grid_index(np.asarray(layout.goal_m))
    if occupied[start] or occupied[goal]:
        return ()
    frontier: list[tuple[float, tuple[int, int, int]]] = [(0.0, start)]
    parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start: None}
    cost = {start: 0.0}
    moves = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            path: list[tuple[float, float, float]] = []
            cursor: tuple[int, int, int] | None = current
            while cursor is not None:
                path.append(tuple(_grid_point(cursor).tolist()))
                cursor = parent[cursor]
            return tuple(reversed(path))
        for delta in moves:
            neighbour = tuple(current[axis] + delta[axis] for axis in range(3))
            if any(neighbour[axis] < 0 or neighbour[axis] >= shape[axis] for axis in range(3)):
                continue
            if occupied[neighbour]:
                continue
            next_cost = cost[current] + _GRID_M
            if next_cost >= cost.get(neighbour, math.inf):
                continue
            cost[neighbour] = next_cost
            parent[neighbour] = current
            heuristic = float(np.linalg.norm(np.asarray(neighbour) - np.asarray(goal))) * _GRID_M
            heapq.heappush(frontier, (next_cost + heuristic, neighbour))
    return ()


def _route_direction_and_side(layout: NavigationLayout) -> tuple[str, str]:
    """Classify the largest signed detour in the goal-relative route frame."""

    start = np.asarray(layout.start_m, dtype=np.float64)
    goal = np.asarray(layout.goal_m, dtype=np.float64)
    delta = goal - start
    denominator = float(np.dot(delta, delta))
    horizontal = delta[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm <= 1.0e-6:
        return "vertical_detour", "above"
    lateral_axis = np.asarray(
        (-horizontal[1], horizontal[0], 0.0), dtype=np.float64
    ) / horizontal_norm
    direction = delta / math.sqrt(denominator)
    vertical_axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    vertical_axis -= float(np.dot(vertical_axis, direction)) * direction
    vertical_norm = float(np.linalg.norm(vertical_axis))
    if vertical_norm > 1.0e-6:
        vertical_axis /= vertical_norm
    largest_lateral = 0.0
    largest_vertical = 0.0
    for point_m in layout.reference_path:
        point = np.asarray(point_m, dtype=np.float64)
        fraction = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
        residual = point - (start + fraction * delta)
        lateral = float(np.dot(residual, lateral_axis))
        vertical = float(np.dot(residual, vertical_axis)) if vertical_norm > 1.0e-6 else 0.0
        if abs(lateral) > abs(largest_lateral):
            largest_lateral = lateral
        if abs(vertical) > abs(largest_vertical):
            largest_vertical = vertical
    if abs(largest_vertical) > abs(largest_lateral):
        return "vertical_detour", "above" if largest_vertical >= 0.0 else "below"
    return "lateral_detour", "left" if largest_lateral >= 0.0 else "right"


def _route_family(layout: NavigationLayout) -> str:
    return _route_direction_and_side(layout)[0]


def _minimum_path_clearance(layout: NavigationLayout, sample_spacing_m: float) -> float:
    if not layout.obstacles or len(layout.reference_path) < 2:
        return math.inf
    return min(
        layout.segment_minimum_clearance(first, second, sample_spacing_m)
        for first, second in zip(layout.reference_path, layout.reference_path[1:])
    )


def generate_layout(
    seed: int,
    difficulty: LayoutDifficulty | int,
    vehicle_radius_m: float,
    margin_m: float,
    endpoint_xy_boundary_buffer_m: float = 0.0,
    obstacle_count_override: int | None = None,
) -> NavigationLayout:
    """Generate one deterministic layout with a hidden feasible path."""

    if (
        vehicle_radius_m <= 0.0
        or margin_m < 0.0
        or endpoint_xy_boundary_buffer_m < 0.0
    ):
        raise ValueError(
            "vehicle_radius_m must be positive; margins and buffers must be non-negative"
        )
    difficulty = LayoutDifficulty(difficulty)
    default_obstacle_count = _OBSTACLE_COUNTS[difficulty]
    obstacle_count = (
        default_obstacle_count
        if obstacle_count_override is None
        else obstacle_count_override
    )
    if obstacle_count < 0 or obstacle_count > len(_SLOT_GEOMETRY):
        raise ValueError(
            f"obstacle_count_override must be between 0 and {len(_SLOT_GEOMETRY)}"
        )
    if difficulty is LayoutDifficulty.OPEN and obstacle_count != 0:
        raise ValueError("OPEN layouts cannot contain obstacles")
    if difficulty is not LayoutDifficulty.OPEN and obstacle_count == 0:
        raise ValueError("non-OPEN layouts must contain at least one obstacle")
    rng = np.random.default_rng(seed)
    clearance_m = vehicle_radius_m + margin_m
    slot_indices = _selected_slot_indices(rng, obstacle_count)

    endpoint_low = _BOUNDS_MIN + clearance_m
    endpoint_high = _BOUNDS_MAX - clearance_m
    xy_buffer = max(clearance_m, endpoint_xy_boundary_buffer_m)
    endpoint_low[:2] = _BOUNDS_MIN[:2] + xy_buffer
    endpoint_high[:2] = _BOUNDS_MAX[:2] - xy_buffer
    endpoint_low[2] = max(endpoint_low[2], _MINIMUM_FLIGHT_HEIGHT_M + _GRID_M)
    if np.any(endpoint_low >= endpoint_high):
        raise ValueError("endpoint boundary buffer leaves no sampling volume")
    for _ in range(128):
        start = _grid_point(_grid_index(rng.uniform(endpoint_low, endpoint_high)))
        goal = _grid_point(_grid_index(rng.uniform(endpoint_low, endpoint_high)))
        if float(np.linalg.norm(goal - start)) < 4.0:
            continue
        if np.any(start < endpoint_low) or np.any(start > endpoint_high):
            continue
        if np.any(goal < endpoint_low) or np.any(goal > endpoint_high):
            continue
        try:
            obstacles = _sample_obstacles(
                rng,
                obstacle_count,
                start,
                goal,
                clearance_m,
                slot_indices=slot_indices,
            )
        except RuntimeError:
            continue
        candidate = NavigationLayout(
            bounds_min_m=tuple(_BOUNDS_MIN.tolist()),
            bounds_max_m=tuple(_BOUNDS_MAX.tolist()),
            start_m=tuple(start.tolist()),
            goal_m=tuple(goal.tolist()),
            obstacles=obstacles,
            reference_path=(),
        )
        if not candidate.point_is_clear(candidate.start_m, clearance_m):
            continue
        if not candidate.point_is_clear(candidate.goal_m, clearance_m):
            continue
        path = _reference_path(candidate, clearance_m)
        measured = NavigationLayout(
            bounds_min_m=candidate.bounds_min_m,
            bounds_max_m=candidate.bounds_max_m,
            start_m=candidate.start_m,
            goal_m=candidate.goal_m,
            obstacles=obstacles,
            reference_path=path,
        )
        physically_clear = measured.path_is_clear(path, clearance_m, sample_spacing_m=0.05)
        direct_clearance = measured.segment_minimum_clearance(
            measured.start_m, measured.goal_m, sample_spacing_m=0.05
        )
        direct_blocked = difficulty is not LayoutDifficulty.OPEN and direct_clearance < clearance_m
        if not direct_blocked:
            detour_direction, detour_side = "direct", "center"
        else:
            detour_direction, detour_side = _route_direction_and_side(measured)
        minimum_clearance = _minimum_path_clearance(measured, sample_spacing_m=0.05)
        narrow = direct_blocked and minimum_clearance <= clearance_m + 0.15
        family = "narrow_passage" if narrow else detour_direction
        if path and physically_clear:
            return NavigationLayout(
                bounds_min_m=candidate.bounds_min_m,
                bounds_max_m=candidate.bounds_max_m,
                start_m=candidate.start_m,
                goal_m=candidate.goal_m,
                obstacles=obstacles,
                reference_path=path,
                route_family=family,
                detour_direction=detour_direction,
                narrow_passage=narrow,
                direct_path_blocked=direct_blocked,
                minimum_route_clearance_m=minimum_clearance,
                route_evidence="grid_nodes_and_0.05_m_segment_samples",
                detour_side=detour_side,
            )
    raise RuntimeError(f"could not generate reachable {difficulty.name.lower()} layout for seed {seed}")
