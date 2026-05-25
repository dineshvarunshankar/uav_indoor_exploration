from __future__ import annotations

from pathlib import Path

import torch
import yaml
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

_spawn_yaml = Path(__file__).resolve().parent.parent / "spawn_zones.yaml"


#load spawn zones
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


def reset_spawn_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    velocity_range: dict[str, tuple[float, float]],
    roll_range: tuple[float, float] = (-0.1, 0.1),
    pitch_range: tuple[float, float] = (-0.1, 0.1),
    yaw_range: tuple[float, float] = (-3.14159, 3.14159),
    spawn_zones: torch.Tensor | None = None,
    z_noise_m: float = 0.5, #Z jitter on random resets (meters), applied on top of zone z
):
    """Reset root pose from spawn_zones.yaml; init uses starling_2 default, later resets randomize."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    spawn_zones = spawn_zones if spawn_zones is not None else global_spawn_zones
    spawn_zones = spawn_zones.to(device=asset.device)

    root_states = asset.data.default_root_state[env_ids].clone()
    n = len(env_ids)

    #initialization
    if env.common_step_counter == 0:
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
        orientations = root_states[:, 3:7]
        velocities = torch.zeros(n, 6, device=asset.device)

    #randomization for subsequent resets
    else:
        #positions
        idx = torch.randint(0, spawn_zones.shape[0], (n,), device=asset.device)
        r = spawn_zones[idx]
        x = math_utils.sample_uniform(r[:,0], r[:,2], (n,), device=asset.device)
        y = math_utils.sample_uniform(r[:,1], r[:,3], (n,), device=asset.device)
        # x = torch.rand(n, device=asset.device) * (r[:,2] - r[:,0]) + r[:,0]
        # y = torch.rand(n, device=asset.device) * (r[:,3] - r[:,1]) + r[:,1]
        z_base = r[:,4]
        z_noise = math_utils.sample_uniform(-z_noise_m, z_noise_m, (n,), device=asset.device)
        z = z_base + z_noise
        positions = torch.stack([x, y, z], dim=-1) + env.scene.env_origins[env_ids]

        #orientations
        roll = math_utils.sample_uniform(roll_range[0], roll_range[1], (n,), device=asset.device)
        pitch = math_utils.sample_uniform(pitch_range[0], pitch_range[1], (n,), device=asset.device)
        yaw = math_utils.sample_uniform(yaw_range[0], yaw_range[1], (n,), device=asset.device)
        orientations = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        #velocities
        range_list = [velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, dtype=torch.float32, device=asset.device)
        velocities = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=asset.device)

    #write to sim
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
