"""Manager configuration for random-goal obstacle-navigation training."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from ..building_inspection.mdp.actions import PX4PositionActionCfg
from ..perception import DisparityCfg
from . import mdp
from .reward_calibration import (
    DENSE_SAFETY_REWARD_WEIGHTS,
    DEPTHNAV_REWARD_WEIGHTS,
    STAGE_ONE_REWARD_WEIGHTS,
)
from .scene import ObstacleNavigationSceneCfg


@configclass
class ActionsCfg:
    px4_waypoint = PX4PositionActionCfg(
        asset_name="robot",
        joint_names=["joint0", "joint1", "joint2", "joint3"],
        prop_body_names=["rotor0", "rotor1", "rotor2", "rotor3"],
        v_max_xy=3.0,
        v_max_z=1.0,
        reach_seconds=0.5,
        randomize=False,
        max_action_delay_steps=0,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        navigation = ObsTerm(func=mdp.policy_observation)

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    @configclass
    class DisparityObservationCfg(ObsGroup):
        disparity = ObsTerm(func=mdp.disparity_observation)

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    @configclass
    class MemoryResetObservationCfg(ObsGroup):
        memory_reset = ObsTerm(func=mdp.memory_reset_observation)

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    @configclass
    class StoppingRiskCfg(ObsGroup):
        stopping_risk = ObsTerm(func=mdp.stopping_risk)

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        navigation = ObsTerm(func=mdp.policy_observation)
        privileged = ObsTerm(func=mdp.privileged_observation)

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    disparity: DisparityObservationCfg = DisparityObservationCfg()
    memory_reset: MemoryResetObservationCfg = MemoryResetObservationCfg()
    stopping_risk: StoppingRiskCfg = StoppingRiskCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Stage-one terms with weights fixed by scripted controls."""

    goal_progress = RewTerm(func=mdp.goal_progress, weight=STAGE_ONE_REWARD_WEIGHTS["goal_progress"])
    goal_reached = RewTerm(func=mdp.goal_reached, weight=STAGE_ONE_REWARD_WEIGHTS["goal_reached"])
    collision_event = RewTerm(func=mdp.collision_event, weight=STAGE_ONE_REWARD_WEIGHTS["collision_event"])
    collision_avoidance = RewTerm(
        func=mdp.collision_avoidance,
        weight=DEPTHNAV_REWARD_WEIGHTS["collision_avoidance"],
    )
    workspace_event = RewTerm(func=mdp.workspace_event, weight=STAGE_ONE_REWARD_WEIGHTS["workspace_event"])
    altitude_margin = RewTerm(func=mdp.altitude_margin, weight=DENSE_SAFETY_REWARD_WEIGHTS["altitude_margin"])
    altitude_violation = RewTerm(func=mdp.altitude_violation, weight=STAGE_ONE_REWARD_WEIGHTS["altitude_violation"])
    action_change = RewTerm(func=mdp.action_change, weight=STAGE_ONE_REWARD_WEIGHTS["action_change"])
    body_rate = RewTerm(func=mdp.body_rate, weight=STAGE_ONE_REWARD_WEIGHTS["body_rate"])
    yaw_acceleration = RewTerm(func=mdp.yaw_acceleration, weight=STAGE_ONE_REWARD_WEIGHTS["yaw_acceleration"])
    yaw_alignment = RewTerm(
        func=mdp.yaw_alignment,
        weight=DEPTHNAV_REWARD_WEIGHTS["yaw_alignment"],
    )


@configclass
class EventCfg:
    randomize_light = EventTerm(
        func=mdp.randomize_dome_light,
        mode="interval",
        interval_range_s=(7.0, 12.0),
        is_global_time=True,
        params={
            "intensity_range": (900.0, 2200.0),
            "color_range": (0.65, 1.0),
        },
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    collision = DoneTerm(func=mdp.collision)
    out_of_bounds = DoneTerm(func=mdp.out_of_bounds)
    goal_reached = DoneTerm(func=mdp.reached_goal)


@configclass
class ObstacleNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Random-goal navigation training environment."""

    scene: InteractiveSceneCfg = ObstacleNavigationSceneCfg(
        num_envs=256,
        env_spacing=25.0,
        filter_collisions=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    disparity: DisparityCfg = DisparityCfg()
    layout_bank_size: int = 256
    training_seed_start: int = 10_000
    held_out_seed_start: int = 1_000_000
    use_held_out_layouts: bool = False
    evaluation_difficulty: int | None = None
    record_episode_records: bool = False
    initial_difficulty: int = 0
    rehearsal_boosts: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    obstacle_count_overrides: tuple[int, int, int, int] = (0, 6, 12, 20)
    goal_tolerance_m: float = 0.5
    yaw_motion_blend_speed_m_s: float = 1.0
    minimum_altitude_m: float = 2.5

    vehicle_radius_m: float = 0.192
    training_clearance_margin_m: float = 0.15
    endpoint_xy_boundary_buffer_m: float = 2.635

    measured_braking_distance_m: float = 1.385
    measured_braking_speed_m_s: float = 3.025
    control_period_s: float = 0.1
    safety_influence_m: float = 4.0
    danger_influence_m: float = 2.443
    depthnav_barrier_scale: float = 2.5
    depthnav_barrier_slope: float = -6.0
    depthnav_yaw_disable_within_m: float = 1.0
    altitude_margin_height_m: float = 1.0
    reward_weights_calibrated: bool = True
    training_ready: bool = True

    def __post_init__(self) -> None:
        self.num_rerenders_on_reset = 1
        self.decimation = 12
        self.episode_length_s = 30.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.viewer.eye = (12.0, 12.0, 10.0)
        self.sim.physx = sim_utils.PhysxCfg()
