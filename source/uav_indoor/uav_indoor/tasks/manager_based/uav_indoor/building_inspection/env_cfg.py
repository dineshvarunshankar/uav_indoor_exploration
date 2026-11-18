"""Scene and manager configuration for the building-inspection task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass

from uav_indoor.assets.starling_2.starling_2 import STARLING2_CFG

from ..inspection import BuildingMissionCfg, MissionStage
from ..perception import INFERENCE_IMAGE_SIZE, YOLOE_MODEL_PATH, DisparityCfg, TofCfg
from .entry_training import EntryTrainingCfg
from .proposals import validate_association_residual
from .site_colliders import SiteColliderCfg, spawn_site
from .site_colliders import enable_on as enable_site_colliders
from .truth import OPENING_TRUTH_COUNT

RENDER_WIDTH, RENDER_HEIGHT = 384, 288
from . import mdp
from .mdp.actions import PX4PositionActionCfg

SITE = "omniverse://airlab-nucleus.andrew.cmu.edu//Public/DTC/ConstructionSite_custom/Collected_ConstructionSite.stage"


@configclass
class SiteUsdFileCfg(sim_utils.UsdFileCfg):
    """Serializable USD spawner configuration with collider settings."""

    func = spawn_site
    site_colliders: SiteColliderCfg = SiteColliderCfg()


@configclass
class BuildingInspectionSceneCfg(InteractiveSceneCfg):
    """Construction site, Starling 2, and the VOXL sensor set."""

    sky = AssetBaseCfg(
        prim_path="/World/Sky",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{SITE}/sky.usd"),
    )
    scene = AssetBaseCfg(
        prim_path="/World/ConstructionSite",
        spawn=SiteUsdFileCfg(usd_path=f"{SITE}/ConstructionSite.stage.usd"),
    )
    robot: ArticulationCfg = STARLING2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=12, update_period=0.0)
    rgb_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/RGB_IMX412",
        spawn=None,
        data_types=["rgb"],
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        update_latest_camera_pose=True,
        update_period=0.2,
    )
    tof_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/ToF_sensor",
        spawn=None,
        data_types=["distance_to_image_plane"],
        width=240,
        height=180,
        update_latest_camera_pose=True,
        update_period=0.03,
    )


@configclass
class ActionsCfg:
    px4_waypoint = PX4PositionActionCfg(
        asset_name="robot",
        joint_names=["joint0", "joint1", "joint2", "joint3"],
        prop_body_names=["rotor0", "rotor1", "rotor2", "rotor3"],
        v_max_xy=3.0,
        v_max_z=1.0,
        randomize=False,
        max_action_delay_steps=0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable observations only."""

        state = ObsTerm(func=mdp.compact_state)
        reference = ObsTerm(func=mdp.compact_reference)
        motion_mode = ObsTerm(func=mdp.motion_mode)
        opening_detection = ObsTerm(func=mdp.opening_detection)
        disparity_freshness = ObsTerm(func=mdp.compact_disparity_freshness)
        tof = ObsTerm(func=mdp.compact_tof)
        previous_action = ObsTerm(func=mdp.previous_action)

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    @configclass
    class DisparityObservationCfg(ObsGroup):
        """Square disparity image kept separate for the learned CNN."""

        disparity = ObsTerm(func=mdp.compact_disparity)

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    @configclass
    class MemoryResetObservationCfg(ObsGroup):
        """Mission event that clears stale actor memory."""

        memory_reset = ObsTerm(func=mdp.memory_reset)

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Adds simulator truth."""

        privileged = ObsTerm(func=mdp.privileged_state)

    policy: PolicyCfg = PolicyCfg()
    disparity: DisparityObservationCfg = DisparityObservationCfg()
    memory_reset: MemoryResetObservationCfg = MemoryResetObservationCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    spawn = EventTerm(func=mdp.reset_to_spawn_zone, mode="reset")
    building = EventTerm(func=mdp.assign_building_target, mode="reset")


@configclass
class RewardsCfg:
    """Safety terms are active in every phase."""

    collision = RewTerm(func=mdp.collision_penalty, weight=-20.0)
    recovery_failed = RewTerm(func=mdp.recovery_failed_penalty, weight=-75.0)
    action_rate = RewTerm(func=mdp.action_rate_penalty, weight=-0.01)

    progress = RewTerm(func=mdp.reference_progress, weight=10.0)
    transit_completion = RewTerm(func=mdp.transit_completion, weight=25.0)
    yaw_progress = RewTerm(func=mdp.reference_yaw_progress, weight=0.05)
    coverage = RewTerm(func=mdp.survey_coverage, weight=2.0)
    crossing = RewTerm(func=mdp.authorized_crossing, weight=10.0)
    exit_route = RewTerm(func=mdp.exit_progress, weight=2.0)
    completion = RewTerm(func=mdp.opening_completion, weight=25.0)
    safe_exit_time = RewTerm(func=mdp.safe_exit_time_bonus, weight=10.0)
    floor = RewTerm(func=mdp.floor_completion, weight=50.0)

    entry_crossing = RewTerm(func=mdp.entry_crossing_bonus, weight=40.0)
    entry_unauthorized = RewTerm(func=mdp.unauthorized_crossing_penalty, weight=-40.0)
    entry_wrong_opening = RewTerm(func=mdp.wrong_opening_penalty, weight=-20.0)


@configclass
class CurriculumCfg:
    stage = CurrTerm(func=mdp.mission_stage)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    mission_complete = DoneTerm(func=mdp.mission_done, time_out=False)
    collision = DoneTerm(func=mdp.collision)
    recovery_failed = DoneTerm(func=mdp.recovery_failed)
    entry_success = DoneTerm(func=mdp.entry_skill_success, time_out=False)
    unauthorized_crossing = DoneTerm(func=mdp.unauthorized_crossing)


@configclass
class BuildingInspectionEnvCfg(ManagerBasedRLEnvCfg):
    scene: BuildingInspectionSceneCfg = BuildingInspectionSceneCfg(num_envs=8, env_spacing=0.0, filter_collisions=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    mission: BuildingMissionCfg = BuildingMissionCfg()
    tof: TofCfg = TofCfg()
    disparity: DisparityCfg = DisparityCfg()
    opening_proposal_path: str | None = None
    proposal_map_version: int = 0
    maximum_association_residual_m: float | None = None
    yoloe_model_path: str = str(YOLOE_MODEL_PATH)
    detector_device: str = "0"
    detector_image_size: int = INFERENCE_IMAGE_SIZE
    detector_period_s: float = 1.0 / 3.0
    detector_confidence: float = 0.10
    detector_backend: str = "prompts"
    track_prediction_grace_s: float = 1.0
    min_openings_for_plane: int = 3

    building_standoff_w: tuple[float, float, float] = (-5.66, 6.0, 12.0)

    approach_box_w: tuple[float, ...] = (-50.0, -35.0, -4.0, 16.0, 8.0, 16.0)
    nominal_person_height_m: float = 1.70
    max_clearance_m: float = 50.0
    max_truth_openings: int = OPENING_TRUTH_COUNT
    truth_match_radius_m: float = 2.0
    tof_grid_shape: tuple[int, int] = (12, 16)
    unauthorized_depth_m: float = 0.25
    crossing_span_m: float = 6.0
    record_episode_results: bool = False
    entry_training: EntryTrainingCfg = EntryTrainingCfg()
    site_colliders: SiteColliderCfg = SiteColliderCfg()

    def __post_init__(self) -> None:
        self.maximum_association_residual_m = validate_association_residual(
            self.maximum_association_residual_m
        )
        if self.proposal_map_version < 0:
            raise ValueError("proposal_map_version must be non-negative")
        if self.site_colliders.enabled:
            enable_site_colliders(self)
        self.decimation = 12
        self.episode_length_s = 300.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.viewer.eye = (8.0, 0.0, 5.0)


@configclass
class EntryCurriculumEnvCfg(BuildingInspectionEnvCfg):
    """The isolated entry episode for the shared mission policy."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.entry_training.enabled = True
        self.mission.facades_per_floor = 1
        self.mission.lock_stage = True
        self.mission.initial_stage = int(MissionStage.FULL)
        self.episode_length_s = 90.0
