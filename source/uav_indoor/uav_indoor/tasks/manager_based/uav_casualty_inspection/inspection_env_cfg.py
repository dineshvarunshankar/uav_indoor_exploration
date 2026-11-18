"""Configuration for the casualty-inspection PPO task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg

from uav_indoor.assets.starling_2.starling_2 import STARLING2_CFG

from . import mdp

MEDIC_USD = f"{ISAAC_NUCLEUS_DIR}/People/Characters/original_male_adult_medical_01/male_adult_medical_01.usd"
WORKER_USD = f"{ISAAC_NUCLEUS_DIR}/People/Characters/original_male_adult_construction_03/male_adult_construction_03.usd"
WORKER_ALT_USD = f"{ISAAC_NUCLEUS_DIR}/People/Characters/original_male_adult_construction_05/male_adult_construction_05.usd"
BUSINESS_USD = f"{ISAAC_NUCLEUS_DIR}/People/Characters/original_female_adult_business_02/female_adult_business_02.usd"
BIPED_SETUP_USD = f"{ISAAC_NUCLEUS_DIR}/People/Characters/Biped_Setup.usd"
PERSON_USDS = (MEDIC_USD, WORKER_USD, BUSINESS_USD, WORKER_ALT_USD, MEDIC_USD) * 2
PERSON_POSITIONS = (
    (16.0, 0.0, 0.0),
    (16.6, 14.43, 0.0),
    (6.07, 26.31, 0.0),
    (-4.33, 13.31, 0.0),
    (-20.48, 14.34, 0.0),
    (-18.97, -0.99, 0.0),
    (-21.76, -17.62, 0.0),
    (-3.63, -14.55, 0.0),
    (8.24, -21.47, 0.0),
    (16.38, -11.47, 0.0),
)


def _person_cfg(name: str, usd_path: str, position: tuple[float, float, float]) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path, scale=(1.0, 1.0, 1.0)),
    )


def _person_proxy_cfg(index: int, position: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/PersonProxy{index}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(position[0], position[1], 0.85)),
        spawn=sim_utils.CapsuleCfg(
            radius=0.30,
            height=1.7,
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


def _static_obstacle_cfg(
    name: str,
    position: tuple[float, float, float],
    shape: str,
) -> RigidObjectCfg:
    common = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        "collision_props": sim_utils.CollisionPropertiesCfg(),
    }
    if shape == "pole":
        spawn = sim_utils.CylinderCfg(
            radius=0.18,
            height=4.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.38)),
            **common,
        )
    elif shape == "crate":
        spawn = sim_utils.CuboidCfg(
            size=(1.2, 1.2, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.40, 0.24, 0.10)),
            **common,
        )
    else:
        spawn = sim_utils.CuboidCfg(
            size=(3.5, 0.25, 1.2),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.55, 0.08)),
            **common,
        )
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        spawn=spawn,
    )


@configclass
class CasualtyInspectionSceneCfg(InteractiveSceneCfg):
    """Plain Isaac scene with a camera, casualties, and kinematic hazards."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1000.0, 1000.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.85, 0.90, 1.0), intensity=1000.0),
    )
    robot: ArticulationCfg = STARLING2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=STARLING2_CFG.init_state.replace(pos=(0.0, 0.0, 2.0)),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
    )
    rgb_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/RGB_IMX412",
        spawn=None,
        data_types=["rgb"],
        width=640,
        height=480,
        update_period=0.1,
    )

    animation_graph = AssetBaseCfg(
        prim_path="/World/Characters/Biped_Setup",
        spawn=sim_utils.UsdFileCfg(usd_path=BIPED_SETUP_USD, visible=False),
    )
    person_0 = _person_cfg("Person0", PERSON_USDS[0], PERSON_POSITIONS[0])
    person_1 = _person_cfg("Person1", PERSON_USDS[1], PERSON_POSITIONS[1])
    person_2 = _person_cfg("Person2", PERSON_USDS[2], PERSON_POSITIONS[2])
    person_3 = _person_cfg("Person3", PERSON_USDS[3], PERSON_POSITIONS[3])
    person_4 = _person_cfg("Person4", PERSON_USDS[4], PERSON_POSITIONS[4])
    person_5 = _person_cfg("Person5", PERSON_USDS[5], PERSON_POSITIONS[5])
    person_6 = _person_cfg("Person6", PERSON_USDS[6], PERSON_POSITIONS[6])
    person_7 = _person_cfg("Person7", PERSON_USDS[7], PERSON_POSITIONS[7])
    person_8 = _person_cfg("Person8", PERSON_USDS[8], PERSON_POSITIONS[8])
    person_9 = _person_cfg("Person9", PERSON_USDS[9], PERSON_POSITIONS[9])
    person_proxies = RigidObjectCollectionCfg(
        rigid_objects={f"person_{index}": _person_proxy_cfg(index, position) for index, position in enumerate(PERSON_POSITIONS)}
    )
    static_obstacles = RigidObjectCollectionCfg(
        rigid_objects={
            "pole_0": _static_obstacle_cfg("Pole0", (4.0, 4.0, 2.0), "pole"),
            "pole_1": _static_obstacle_cfg("Pole1", (-4.0, 4.0, 2.0), "pole"),
            "pole_2": _static_obstacle_cfg("Pole2", (-4.0, -4.0, 2.0), "pole"),
            "pole_3": _static_obstacle_cfg("Pole3", (4.0, -4.0, 2.0), "pole"),
            "crate_0": _static_obstacle_cfg("Crate0", (1.5, 6.0, 0.5), "crate"),
            "crate_1": _static_obstacle_cfg("Crate1", (-6.0, -1.5, 0.5), "crate"),
            "barrier_0": _static_obstacle_cfg("Barrier0", (0.0, 9.0, 0.6), "barrier"),
            "barrier_1": _static_obstacle_cfg("Barrier1", (0.0, -9.0, 0.6), "barrier"),
        }
    )
    dynamic_vehicle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DynamicVehicle",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(6.0, 0.0, 0.5)),
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.25)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )
    dynamic_drone = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DynamicDrone",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.0, 0.0, 2.5)),
        spawn=sim_utils.SphereCfg(
            radius=0.45,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


@configclass
class ActionsCfg:
    px4_velocity = mdp.PX4VelocityActionCfg(
        asset_name="robot",
        v_max_xy=3.0,
        v_max_z=1.0,
        randomize=False,
        max_action_delay_steps=0,
        motor_tau=0.02,
    )


@configclass
class ObservationsCfg:
    @configclass
    class DepthCfg(ObsGroup):
        depth = ObsTerm(func=mdp.temporal_depth)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ActorCfg(ObsGroup):
        tracks = ObsTerm(func=mdp.pose_tracks)
        reference = ObsTerm(
            func=mdp.local_reference_body,
            noise=GaussianNoiseCfg(mean=0.0, std=0.08),
        )
        vehicle = ObsTerm(
            func=mdp.vehicle_state,
            noise=GaussianNoiseCfg(mean=0.0, std=0.03),
        )
        previous_action = ObsTerm(func=mdp.previous_action)
        phase = ObsTerm(func=mdp.mission_phase)
        mission = ObsTerm(func=mdp.mission_status)
        freshness = ObsTerm(func=mdp.perception_freshness)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        scene_state = ObsTerm(func=mdp.privileged_scene_state)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    depth: DepthCfg = DepthCfg()
    actor: ActorCfg = ActorCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventsCfg:
    reset_robot = EventTerm(func=mdp.reset_robot, mode="reset")
    reset_mission = EventTerm(func=mdp.reset_mission, mode="reset")
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    wind = None


@configclass
class RewardsCfg:
    progress = RewTerm(func=mdp.local_reference_progress, weight=8.0)
    tracking = RewTerm(func=mdp.reference_tracking_exp, weight=2.0, params={"std": 12.0})
    heading = RewTerm(func=mdp.reference_heading_exp, weight=1.0, params={"std": 1.2})
    clearance = RewTerm(func=mdp.clearance_ttc, weight=-2.0)
    phase_progress = RewTerm(func=mdp.mission_phase_progress, weight=10.0)
    visibility = RewTerm(func=mdp.target_visibility, weight=2.0)
    face = RewTerm(func=mdp.face_acquisition, weight=3.0)
    staging = RewTerm(func=mdp.staging_accuracy, weight=2.0)
    recording = RewTerm(func=mdp.recording_quality, weight=3.0)
    arc_framing = RewTerm(func=mdp.arc_framing_lower_quantile, weight=1.5)
    completion = RewTerm(func=mdp.mission_completion, weight=5.0)
    failure = RewTerm(func=mdp.mission_failure, weight=-100.0)
    shield = RewTerm(func=mdp.shield_intervention, weight=-0.2)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.04)
    collision = RewTerm(
        func=mdp.collision,
        weight=-100.0,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
    )
    out_of_bounds = RewTerm(func=mdp.out_of_bounds, weight=-100.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    collision = DoneTerm(
        func=mdp.collision,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
    )
    out_of_bounds = DoneTerm(func=mdp.out_of_bounds)
    mission_failed = DoneTerm(func=mdp.mission_failed)


@configclass
class CurriculumCfg:
    stage = CurrTerm(
        func=mdp.staged_mission_curriculum,
        params={"success_rate": 0.6, "minimum_transitions": 20_000},
    )


@configclass
class CasualtyInspectionEnvCfg(ManagerBasedRLEnvCfg):
    """10 Hz model-in-loop PPO task."""

    scene: CasualtyInspectionSceneCfg = CasualtyInspectionSceneCfg(
        num_envs=1, env_spacing=120.0, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    midas_model_path: str = "/home/ubuntu/starling/AI_VOXL2/onnx/MiDaS/midas.onnx"
    pose_model_path: str = "/home/ubuntu/yolo26n-pose.pt"
    pose_device: str = "0"
    photometric_randomization: bool = True
    frame_drop_probability: float = 0.03
    max_casualties: int = 10
    max_hazards: int = 2
    gps_uncertainty_m: tuple[float, float] = (1.0, 3.0)
    curriculum_stage: int = 0
    auto_curriculum: bool = True

    def __post_init__(self) -> None:
        self.decimation = 12
        self.episode_length_s = 360.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (12.0, 12.0, 8.0)
