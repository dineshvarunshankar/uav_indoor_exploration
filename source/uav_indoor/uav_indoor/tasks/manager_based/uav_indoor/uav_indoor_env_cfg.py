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
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg

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
    # scene: single SHARED instance (not per-env) so the thousands of colliders are paid for
    # once instead of x num_envs. All robots fly in this one world; env_origins are ~0
    # (env_spacing=0), so spawn-zone / opening coords are used directly as world coords.
    scene = AssetBaseCfg(
        prim_path="/World/ConstructionSite",
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
    # contact sensor on all robot bodies -> used for collision termination.
    # NOTE: requires the scene USD (walls/floor) to have collision geometry, else no contacts are reported.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
    )
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
        prop_body_names=["rotor0", "rotor1", "rotor2", "rotor3"],
        v_max_xy=3.0,
        v_max_z=1.0,
        yaw_max=math.pi,
        motor_scale=1.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ProprioCfg(ObsGroup):
        """Observations for policy group."""

        # body (noise std mirrors real EKF/IMU/perception error so the policy can't
        # rely on perfectly clean state at deploy time)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.07))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.02))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoiseCfg(mean=0.0, std=0.02))
        # root_pos_w removed: absolute world position hurts generalization and is large/unnormalized.
        # target_offset_body already gives the policy the relative bearing+distance it needs.
        #base_pos_z = ObsTerm(func=mdp.base_pos_z)

        # rotors: joint_vel_rel removed. Per-rotor RPM is not a reliable real-time
        # observation on the real Starling 2, so training on it is a sim-only crutch.
        #joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)

        #prev actions (policy's own output: exact, no noise)
        last_action = ObsTerm(func=mdp.last_action)

        #opening target (from perception/localization: largest real error source)
        target_offset_body = ObsTerm(func=mdp.target_offset_body, noise=GaussianNoiseCfg(mean=0.0, std=0.10))


        def __post_init__(self) -> None:
            self.enable_corruption = True
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
                "x": (-0.5, 0.5), #(-3.0, 3.0),      # 3 m/s forward/backward
                "y": (-0.5, 0.5), #(-3.0, 3.0),      # 3 m/s left/right
                "z": (-0.25, 0.25), #(-1.0, 1.0),      # 1 m/s up/down
                "roll": (-0.5, 0.5), #(-1.745, 1.745),   # 100 deg/s roll rate
                "pitch": (-0.5, 0.5), #(-1.745, 1.745),  # 100 deg/s pitch rate
                "yaw": (-0.2, 0.2), #(-0.698, 0.698),    # 40 deg/s yaw rate
            },
        },
    )

    # Sim-to-real: randomize total mass each episode (T/W spread; pairs with the
    # per-env thrust-scale + motor-lag randomization in the PX4 velocity action term)
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
    # (2) Failure penalty: only real failures (collision), NOT time_out or reached_opening success
    terminating = RewTerm(
        func=mdp.is_terminated_term, weight=-10.0,
        params={"term_keys": ["collision"]},
    )
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
    # progress is the primary potential-function driver (telescopes to distance closed)
    progress_to_opening = RewTerm(func=mdp.progress_to_opening, weight=6.0)
    # absolute-distance shaping kept small so it cannot dominate progress / encourage camping
    distance_to_opening_exp = RewTerm(
        func=mdp.distance_to_opening_exp, weight=1.0,
        params={"std": 3.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    heading_to_opening_exp = RewTerm(
        func=mdp.heading_to_opening_exp, weight=2.0,
        params={"std": 0.8, "asset_cfg": SceneEntityCfg("robot")},
    )
    # one-time success bonus: episode ends on success (see TerminationsCfg.reached_opening),
    # so this fires once. Reward manager scales by dt (1/60), so weight 300 -> ~+5 effective.
    at_opening = RewTerm(
        func=mdp.at_opening, weight=100.0,
        params={"success_radius": 0.7, "asset_cfg": SceneEntityCfg("robot")},
    )
    height_to_opening = RewTerm(
        func=mdp.height_error_to_opening_l2, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    body_axis_aligned_to_opening = RewTerm(
    func=mdp.body_axis_aligned_to_opening,
    weight=1.0,
    params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Failure: any collision (drone body/rotors hitting walls, floor, obstacles).
    # threshold in Newtons; tune up if light grazes should be ignored.
    collision = DoneTerm(
        func=mdp.illegal_contact,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
    )
    # (3) Success: reached the opening. Ends the episode so the policy cannot camp at the goal.
    # success_radius must match the at_opening reward so the bonus fires on the terminal step.
    reached_opening = DoneTerm(
        func=mdp.reached_opening,
        params={"success_radius": 0.7, "asset_cfg": SceneEntityCfg("robot")},
    )


##
# Environment configuration
##


@configclass
class UavIndoorEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    # env_spacing=0: shared world, all envs in one coordinate frame; filter_collisions keeps
    # robots from different envs from physically colliding with each other (each still hits the scene).
    scene: UavIndoorSceneCfg = UavIndoorSceneCfg(num_envs=2, env_spacing=0.0, filter_collisions=True)
    # Basic settings 
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2 #how many physics sub-steps per simulation step
        self.episode_length_s = 30  # ~3 m/s cap -> ~90 m reachable; safe now that success ends the episode
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120 #simulation time step; with decimation as 2, the actual time step is 1/120 * 2 = 1/60 ~60Hz
        self.sim.render_interval = self.decimation