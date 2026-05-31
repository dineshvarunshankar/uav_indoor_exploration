# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
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
from isaaclab.sensors import TiledCameraCfg

from . import mdp

##
# custom configs
##

from uav_indoor.assets.starling_2.starling_2 import STARLING2_CFG


##
# Scene definition
##
sky_usd = "omniverse://airlab-nucleus.andrew.cmu.edu//Public/DTC/ConstructionSite_custom/Collected_ConstructionSite.stage/sky.usd"
scene_usd = "omniverse://airlab-nucleus.andrew.cmu.edu//Public/DTC/ConstructionSite_custom/Collected_ConstructionSite.stage/ConstructionSite.stage.usd"

@configclass
class UavIndoorSceneCfg(InteractiveSceneCfg):
    """Configuration for a ConstructionSite scene."""

    #sky
    sky = AssetBaseCfg(
        prim_path="/World/Sky",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            #rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(usd_path=sky_usd),
    )
    # scene
    scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment",
        # prim_path="/World/ConstructionSite",   # single instance
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(usd_path=scene_usd),
    )

    # # ground plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    # )

    # robot
    robot: ArticulationCfg = STARLING2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    #tof camera ModalAI VOXL 2 ToF (M0178)
    tof_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/ToF_sensor", 
        spawn=None, #already defined in USD
        width=240,
        height=180,
        data_types=["distance_to_camera"],
        update_period=0.03,
        depth_clipping_behavior="zero",
    )

    # lights
    # dome_light = AssetBaseCfg(
    #     prim_path="/World/DomeLight",
    #     spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    # )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["joint0", "joint1", "joint2", "joint3"], scale=1.0)
    px4_velocity = mdp.PX4VelocityActionCfg(
        asset_name="robot", 
        joint_names=["joint0", "joint1", "joint2", "joint3"], 
        v_max_xy=3.0,
        v_max_z=1.0,
        yaw_max = math.pi, #yaw_rate_max=2.618,
        motor_scale=1.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ProprioCfg(ObsGroup):
        """Observations for policy group."""

        # body
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        root_pos_w = ObsTerm(func=mdp.root_pos_w)
        #base_pos_z = ObsTerm(func=mdp.base_pos_z)

        #rotors
        #joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        #prev actions
        last_action = ObsTerm(func=mdp.last_action)

        #opening target
        target_offset_body = ObsTerm(func=mdp.target_offset_body)


        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # @configclass
    # class TofCfg(ObsGroup):
    #     tof_depth = ObsTerm(
    #         func=mdp.image,
    #         params={
    #             "sensor_cfg": SceneEntityCfg("tof_camera"),
    #             "data_type": "distance_to_camera",
    #             "normalize": True,
    #         }
    #     )
    #     def __post_init__(self) -> None:
    #         self.enable_corruption = False
    #         self.concatenate_terms = True
    # observation groups
    proprio: ProprioCfg = ProprioCfg()
    #tof: TofCfg = TofCfg()


@configclass
class EventCfg:
    """Reset events."""

     #Randomize drone pose (position/orientation/velocity) around defaults in starling_2.py
    reset_base = EventTerm(
        func=mdp.reset_spawn_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # Position is defined usingspawn_zones.yaml inside the function
            # Orientations (radians)
            "roll_range": (-0.1745, 0.1745),   # +/- 10 degrees
            "pitch_range": (-0.1745, 0.1745),  # +/- 10 degrees
            "yaw_range": (-3.14159, 3.14159),  # Full 360 degrees
            
            # Velocities 
            "velocity_range": {
                "x": (-3.0, 3.0),      # 3 m/s forward/backward
                "y": (-3.0, 3.0),      # 3 m/s left/right
                "z": (-1.0, 1.0),      # 1 m/s up/down
                "roll": (-1.745, 1.745),   # 100 deg/s roll rate
                "pitch": (-1.745, 1.745),  # 100 deg/s pitch rate
                "yaw": (-0.698, 0.698),    # 40 deg/s yaw rate
            },
        },
    )

    # Propellers
    # Rotors 0 and 1 (Counter-Clockwise - Positive Spin)
    reset_rotor_joints_ccw = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint0", "joint1"]), 
            "position_range": (0.0, 0.0),
            "velocity_range": (0,0),#(900.0, 1100.0),
        },
    )

    # Rotors 2 and 3 (Clockwise - Negative Spin)
    reset_rotor_joints_cw = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint2", "joint3"]), 
            "position_range": (0.0, 0.0),
            "velocity_range": (0,0),#(-1100.0, -900.0), 
        },
    )

    # reset_pole_position = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
    #         "position_range": (-0.25 * math.pi, 0.25 * math.pi),
    #         "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
    #     },
    # )
    # In EventCfg:
    assign_openings_on_reset = EventTerm(
        func=mdp.reset_opening_target,
        mode="reset",
    )

    assign_openings_on_startup = EventTerm(
        func=mdp.reset_opening_target,
        mode="startup",
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-10.0)
    # (3) Primary task: keep pole upright
    # pole_pos = RewTerm(
    #     func=mdp.joint_pos_target_l2,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    # )
    # # (4) Shaping tasks: lower cart velocity
    # cart_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.01,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    # )
    # # (5) Shaping tasks: lower pole angular velocity
    # pole_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-0.005,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    # )
    # hover near target height (world z; tune to match spawn ~0.5–1.0 m)
    # height = RewTerm(
    #     func=mdp.base_height_l2,
    #     weight=-2.0,
    #     params={"target_height": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    # )
    # penalize tilt (upright)
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # penalize vertical bobbing and wobble
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
    # smooth control
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.02)

    #REWARDS.PY
    progress_to_opening = RewTerm(func=mdp.progress_to_opening, weight=3.0)
    distance_to_opening_exp = RewTerm(
        func=mdp.distance_to_opening_exp, weight=15.0,
        params={"std": 3.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    heading_to_opening_exp = RewTerm(
        func=mdp.heading_to_opening_exp, weight=2.0,
        params={"std": 0.8, "asset_cfg": SceneEntityCfg("robot")},
    )
    at_opening = RewTerm(
        func=mdp.at_opening, weight=10.0,
        params={"success_radius": 1.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    height_to_opening = RewTerm(
        func=mdp.height_error_to_opening_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    # )
    # hit ground / too low (tune minimum_height for your spawn height)
    low_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.15, "asset_cfg": SceneEntityCfg("robot")},
    )
    # flipped / too tilted (~57° if limit_angle=1.0 rad)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )


##
# Environment configuration
##


@configclass
class UavIndoorEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: UavIndoorSceneCfg = UavIndoorSceneCfg(num_envs=2, env_spacing=6.0, filter_collisions=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2 #how many physics sub-steps per simulation step
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120 #simulation time step; with decimation as 2, the actual time step is 1/120 * 2 = 1/60 ~60Hz
        self.sim.render_interval = self.decimation