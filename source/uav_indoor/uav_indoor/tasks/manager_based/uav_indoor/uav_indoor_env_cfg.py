# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg

from . import mdp


from uav_indoor.assets.starling_2.starling_2 import STARLING2_CFG


sky_usd = "omniverse://airlab-nucleus.andrew.cmu.edu//Public/DTC/ConstructionSite_custom/Collected_ConstructionSite.stage/sky.usd"
scene_usd = "omniverse://airlab-nucleus.andrew.cmu.edu//Public/DTC/ConstructionSite_custom/Collected_ConstructionSite.stage/ConstructionSite.stage.usd"

@configclass
class UavIndoorSceneCfg(InteractiveSceneCfg):
    """Configuration for a ConstructionSite scene."""

    sky = AssetBaseCfg(
        prim_path="/World/Sky",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(usd_path=sky_usd),
    )
    scene = AssetBaseCfg(
        prim_path="/World/ConstructionSite",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(usd_path=scene_usd),
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(240.0, 240.0)),
    )

    robot: ArticulationCfg = STARLING2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
    )
    tof_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/ToF_sensor", 
        spawn=None,
        width=240,
        height=180,
        data_types=["distance_to_camera"],
        update_period=0.03,
        depth_clipping_behavior="zero",
    )





@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    px4_waypoint = mdp.PX4WaypointActionCfg(
        asset_name="robot",
        joint_names=["joint0", "joint1", "joint2", "joint3"],
        prop_body_names=["rotor0", "rotor1", "rotor2", "rotor3"],
        v_max_xy=3.0,
        v_max_z=1.0,
        yaw_max=math.pi,
        motor_scale=1.0,
        randomize=False,
        max_action_delay_steps=0
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the building-inspection MDP."""

    @configclass
    class ProprioCfg(ObsGroup):
        """Deployment-visible actor observations."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.07))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.02))
        rotation = ObsTerm(func=mdp.base_rotation_6d, noise=GaussianNoiseCfg(mean=0.0, std=0.02))
        last_action = ObsTerm(func=mdp.last_action)
        mission_reference = ObsTerm(
            func=mdp.mission_reference_observation,
            noise=GaussianNoiseCfg(mean=0.0, std=0.02),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        disparity_history = ObsTerm(func=mdp.disparity_history_observation)
        tof_corridor = ObsTerm(func=mdp.tof_corridor_observation)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged mission truth for the value function only."""

        mission_privileged = ObsTerm(
            func=mdp.mission_privileged_observation,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Reset events."""

    reset_base = EventTerm(
        func=mdp.reset_spawn_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "roll_range": (-0.1745, 0.1745),   # +/- 10 degrees
            "pitch_range": (-0.1745, 0.1745),  # +/- 10 degrees
            "yaw_range": (-3.14159, 3.14159),  # Full 360 degrees
            
            "spawn_zones": mdp.exterior_building_spawn_zones,
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.25, 0.25),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    reset_rotor_joints_ccw = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint0", "joint1"]), 
            "position_range": (0.0, 0.0),
            "velocity_range": (0,0),
        },
    )

    reset_rotor_joints_cw = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint2", "joint3"]), 
            "position_range": (0.0, 0.0),
            "velocity_range": (0,0),
        },
    )

    reset_building_mission = EventTerm(
        func=mdp.reset_building_mission,
        mode="reset",
    )

    reset_building_mission_on_startup = EventTerm(
        func=mdp.reset_building_mission,
        mode="startup",
    )


@configclass
class RewardsCfg:
    """Phase-aware rewards for full building inspection."""

    terminating = RewTerm(
        func=mdp.is_terminated_term,
        weight=-10.0,
        params={"term_keys": ["collision"]},
    )
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.02)

    clearance_safety = RewTerm(
        func=mdp.clearance_safety_reward,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "margin_m": 1.0},
    )
    mission_progress = RewTerm(
        func=mdp.mission_progress_reward,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    survey_coverage = RewTerm(
        func=mdp.survey_coverage_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    opening_validation = RewTerm(
        func=mdp.opening_validation_reward,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    entry_exit_recovery = RewTerm(
        func=mdp.entry_exit_recovery_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    person_observation = RewTerm(
        func=mdp.person_observation_reward,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    building_complete = RewTerm(
        func=mdp.building_complete_reward,
        weight=20.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the building-inspection MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    collision = DoneTerm(
        func=mdp.illegal_contact,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
    )
    building_done = DoneTerm(
        func=mdp.building_mission_done,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )




@configclass
class UavIndoorEnvCfg(ManagerBasedRLEnvCfg):
    scene: UavIndoorSceneCfg = UavIndoorSceneCfg(num_envs=2, env_spacing=0.0, filter_collisions=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 180
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation