from __future__ import annotations

from pathlib import Path
import torch
import yaml
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from .opening_targets import assign_random_opening_targets

_spawn_yaml = Path(__file__).resolve().parent.parent / "spawn_zones.yaml"


def load_spawn_zones(path: Path | None = None) -> torch.Tensor:
    """Load spawn zones as (N, 5) tensor: x_min, y_min, x_max, y_max, z in meters."""
    path = path or _spawn_yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    rows = [r[:5] for r in data["zones"]]
    zones = torch.tensor(rows, dtype=torch.float32)
    assert torch.all(zones[:, 0] < zones[:, 2]) and torch.all(zones[:, 1] < zones[:, 3])
    return zones


global_spawn_zones = load_spawn_zones()

exterior_building_spawn_zones = torch.tensor(
    [
        [30.0, -110.0, 65.0, -98.0, 3.0],
        [86.0, -60.0, 98.0, -20.0, 3.0],
        [20.0, 28.0, 65.0, 40.0, 3.0],
    ],
    dtype=torch.float32,
)


def reset_spawn_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    velocity_range: dict[str, tuple[float, float]],
    roll_range: tuple[float, float] = (-0.1, 0.1),
    pitch_range: tuple[float, float] = (-0.1, 0.1),
    yaw_range: tuple[float, float] = (-3.14159, 3.14159),
    spawn_zones: torch.Tensor | None = None,
    z_noise_m: float = 0.5,
):
    """Reset root pose from spawn_zones.yaml; randomized on every reset, including the first."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    spawn_zones = spawn_zones if spawn_zones is not None else global_spawn_zones
    spawn_zones = spawn_zones.to(device=asset.device)

    n = len(env_ids)

    idx = torch.randint(0, spawn_zones.shape[0], (n,), device=asset.device)
    r = spawn_zones[idx]
    x = math_utils.sample_uniform(r[:,0], r[:,2], (n,), device=asset.device)
    y = math_utils.sample_uniform(r[:,1], r[:,3], (n,), device=asset.device)
    z_base = r[:,4]
    z_noise = math_utils.sample_uniform(-z_noise_m, z_noise_m, (n,), device=asset.device)
    z = z_base + z_noise
    positions = torch.stack([x, y, z], dim=-1) + env.scene.env_origins[env_ids]

    roll = math_utils.sample_uniform(roll_range[0], roll_range[1], (n,), device=asset.device)
    pitch = math_utils.sample_uniform(pitch_range[0], pitch_range[1], (n,), device=asset.device)
    yaw = math_utils.sample_uniform(yaw_range[0], yaw_range[1], (n,), device=asset.device)
    orientations = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    range_list = [velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, dtype=torch.float32, device=asset.device)
    velocities = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device)

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def reset_opening_target(env, env_ids, openings=None):
    """Event callback: sample new opening goals (mode='reset' or 'startup')."""
    assign_random_opening_targets(env, env_ids, openings)
