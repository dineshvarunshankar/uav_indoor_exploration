import yaml
from pathlib import Path
import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv

_openings_yaml = Path(__file__).resolve().parent.parent / "openings.yaml"

def load_openings(path: Path | None = None) -> torch.Tensor:
    """Load openings as (N, 3) tensor: x, y, z in meters."""
    path = path or _openings_yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return torch.tensor(data["openings"], dtype=torch.float32)


OPENINGS = load_openings()  


def get_opening_targets(env: ManagerBasedEnv) -> torch.Tensor:
    """(num_envs, 3) goal positions; allocates zeros until assign runs."""
    if not hasattr(env, "opening_target_env"):
        env.opening_target_env = torch.zeros(env.num_envs, 3, device=env.device)
    return env.opening_target_env


def assign_random_opening_targets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    openings: torch.Tensor | None = None,
) -> None:
    if env_ids is None or env_ids == slice(None):
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device)
    """Write a random opening from the catalog into opening_target_env[env_ids]."""
    openings = (openings or OPENINGS).to(device=env.device)
    targets = get_opening_targets(env)
    idx = torch.randint(0, openings.shape[0], (len(env_ids),), device=env.device)
    targets[env_ids] = openings[idx]
    _sync_progress_baseline(env, env_ids)  # see below


def _sync_progress_baseline(env, env_ids):
    """Reset progress reward baseline after goal changes."""
    if not hasattr(env, "_prev_dist_to_opening"):
        env._prev_dist_to_opening = torch.zeros(env.num_envs, device=env.device)
    asset = env.scene["robot"]
    robot_pos = robot_position_env(env, asset)[env_ids]
    env._prev_dist_to_opening[env_ids] = torch.norm(
        get_opening_targets(env)[env_ids] - robot_pos, dim=-1
    )


def robot_position_env(env: ManagerBasedEnv, asset: Articulation) -> torch.Tensor:
    return asset.data.root_pos_w - env.scene.env_origins


def offset_to_opening_env(
    env: ManagerBasedEnv, asset: Articulation
) -> torch.Tensor:
    """Vector from robot to goal in env-local frame: (num_envs, 3)."""
    return get_opening_targets(env) - robot_position_env(env, asset)


def offset_to_opening_body(
    env: ManagerBasedEnv, asset: Articulation
) -> torch.Tensor:
    """Same vector in body frame: (num_envs, 3)."""
    delta_env = offset_to_opening_env(env, asset)
    return math_utils.quat_apply_inverse(asset.data.root_quat_w, delta_env)