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

from . import mdp

##
# custom configs
##

from uav_indoor.assets.starling_2.starling_2 import STARLING2_CFG


##
# Scene definition
##
scene_usd = "omniverse://airlab-nucleus.andrew.cmu.edu/Public/AirStack/scenes/urban/abandoned_city/After_the_rain...usdz"

@configclass
class UavIndoorSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""
    # room
    environment = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(usd_path=scene_usd),
    )

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = STARLING2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["joint0", "joint1", "joint2", "joint3"], scale=1.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # body
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_pos_z = ObsTerm(func=mdp.base_pos_z)

        #rotors
        #joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        #prev actions
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events."""

     #Randomize drone pose (position/orientation/velocity) around defaults in starling_2.py
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # Props: no random angle/spin at reset
    reset_rotor_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint0", "joint1", "joint2", "joint3"]),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
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
    height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    # penalize tilt (upright)
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # penalize vertical bobbing and wobble
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # smooth control
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


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
    scene: UavIndoorSceneCfg = UavIndoorSceneCfg(num_envs=64, env_spacing=6.0, filter_collisions=True)
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