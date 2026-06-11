# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from . import mdp

from uav_indoor.assets.starling_2.starling_2 import STARLING2_CFG


##
# Scene definition
##


@configclass
class UavHoverSceneCfg(InteractiveSceneCfg):
    """Hover scene: ground plane + Starling 2."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2000.0),
    )

    # robot spawned ~1 m above ground (reset event randomizes around this)
    robot: ArticulationCfg = STARLING2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=STARLING2_CFG.init_state.replace(pos=(0.0, 0.0, 0.1)),
    )

    # contact sensor - crash termination on ground/self contact
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """PX4 velocity action term + scaling."""

    px4_velocity = mdp.PX4VelocityActionCfg(
        asset_name="robot",
        joint_names=["joint0", "joint1", "joint2", "joint3"],
        prop_body_names=["rotor0", "rotor1", "rotor2", "rotor3"],
        v_max_xy=3.0,
        v_max_z=1.0,
        yaw_max=math.pi,
        motor_scale=1.0,
        # randomize=True, #motor_tau, thrust scale, action delay
        # max_action_delay_steps=2,
        randomize=False,
        max_action_delay_steps=0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ProprioCfg(ObsGroup):

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoiseCfg(mean=0.0, std=0.05))
        xy_error = ObsTerm(func=mdp.hover_xy_error, noise=GaussianNoiseCfg(mean=0.0, std=0.05))

        last_action = ObsTerm(func=mdp.last_action)
        # commanded - current height (m)
        height_error = ObsTerm(func=mdp.hover_height_error, noise=GaussianNoiseCfg(mean=0.0, std=0.05))

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()


@configclass
class EventCfg:
    """Reset / startup events (incl. domain randomization for sim-to-real)."""

    # sample a new commanded hover height each episode (and at startup so obs is valid)
    set_hover_target_startup = EventTerm(
        func=mdp.reset_hover_target,
        mode="startup",
        # params={"height_range": (0.75, 5.0)},
        params={"height_range": (1.0, 5.0)},
    )
    set_hover_target_reset = EventTerm(
        func=mdp.reset_hover_target,
        mode="reset",
        # params={"height_range": (0.75, 5.0)},
        params={"height_range": (1.0, 5.0)},
    )

    # randomized initial pose/velocity around the spawn (offsets from default root state)
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                # "z": (0.1, 0.5),            # spawn
                # "roll": (-0.1745, 0.1745),  
                # "pitch": (-0.1745, 0.1745),
                # "yaw": (-3.14159, 3.14159),
                "z": (0.0, 0.1),              # start at the 1 m default spawn height
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {
                # "x": (-0.5, 0.5),
                # "y": (-0.5, 0.5),
                # "z": (-0.25, 0.25),
                # "roll": (-0.5, 0.5),
                # "pitch": (-0.5, 0.5),
                # "yaw": (-0.2, 0.2),
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    record_episode_xy_ref = EventTerm(
        func=mdp.record_episode_xy_ref,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # sim-to-real: per-episode total-mass spread (pairs with motor-lag / thrust-scale
    # randomization that lives in the PX4 velocity action term)
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.8, 1.2),
            # "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    record_episode_yaw_ref = EventTerm(
    func=mdp.record_episode_yaw_ref,
    mode="reset",
    params={"asset_cfg": SceneEntityCfg("robot")},
    )
    


@configclass
class RewardsCfg:
    """Reward terms for the hover MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=0.1)

    # primary task: be at the commanded height
    height_tracking = RewTerm(
        func=mdp.hover_height_tracking, weight=10.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    xy_tracking = RewTerm(
        func=mdp.episode_xy_tracking,
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    # station-keeping + smooth, level, calm flight
    horizontal_velocity = RewTerm(func=mdp.horizontal_velocity_l2, weight=-0.01)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.001)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.001)
    base_ang_vel = RewTerm(func=mdp.base_ang_vel_l2, weight=-0.001)
    # ang_vel_xy 
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)
    episode_yaw = RewTerm(
    func=mdp.episode_yaw_tracking,
    weight=5.0,
    params={"std": 1.0, "asset_cfg": SceneEntityCfg("robot")},)
    crash_penalty = RewTerm(
        func=mdp.illegal_contact,
        weight=-10.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
        },
    )
    terminating = RewTerm(
        func=mdp.is_terminated_term, weight=-10.0,
        params={"term_keys": ["crash", "flipped"]},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    crash = DoneTerm(
        func=mdp.illegal_contact,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
    )
    flipped = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},  # ~57 deg
    )

##
# Environment configuration
##


@configclass
class UavHoverEnvCfg(ManagerBasedRLEnvCfg):
    # Scene: per-env spacing with a shared ground plane; many parallel envs
    scene: UavHoverSceneCfg = UavHoverSceneCfg(num_envs=1024, env_spacing=2.5)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 60.0
        # viewer
        self.viewer.eye = (4.0, 0.0, 3.0)
        # simulation: 1/120 * decimation 2 -> ~60 Hz control
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
