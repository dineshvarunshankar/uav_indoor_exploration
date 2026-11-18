"""Isaac scene with a fixed pool of visible kinematic colliders."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass

from uav_indoor.assets.starling_2.starling_2 import STARLING2_CFG

from .layouts import ObstacleSpec, obstacle_slot_specs

_OBSTACLE_COLORS = (
    (0.55, 0.34, 0.16),
    (0.24, 0.42, 0.58),
    (0.48, 0.48, 0.43),
    (0.62, 0.49, 0.25),
    (0.31, 0.52, 0.36),
)


def _obstacle_cfg(index: int, spec: ObstacleSpec) -> RigidObjectCfg:
    common = {
        "visual_material": sim_utils.PreviewSurfaceCfg(
            diffuse_color=_OBSTACLE_COLORS[index % len(_OBSTACLE_COLORS)]
        ),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        "collision_props": sim_utils.CollisionPropertiesCfg(),
    }
    if spec.family == "post":
        spawn = sim_utils.CylinderCfg(
            radius=spec.size_m[0] * 0.5,
            height=spec.size_m[2],
            **common,
        )
    else:
        spawn = sim_utils.CuboidCfg(size=spec.size_m, **common)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Obstacle{index:02d}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=spec.center_m),
        spawn=spawn,
    )


@configclass
class ObstacleNavigationSceneCfg(InteractiveSceneCfg):
    """Flat training volume, production aircraft, camera, and colliders."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1000.0, 1000.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1500.0),
    )
    robot: ArticulationCfg = STARLING2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=STARLING2_CFG.init_state.replace(pos=(-8.0, 0.0, 3.0)),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=12,
        update_period=0.0,
        track_air_time=False,
    )
    rgb_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/RGB_IMX412",
        spawn=None,
        data_types=["rgb", "distance_to_image_plane"],
        width=384,
        height=288,
        update_latest_camera_pose=True,
        update_period=0.2,
    )
    obstacles = RigidObjectCollectionCfg(
        rigid_objects={
            f"obstacle_{index:02d}": _obstacle_cfg(index, spec)
            for index, spec in enumerate(obstacle_slot_specs())
        }
    )
