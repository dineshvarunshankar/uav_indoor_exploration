# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

    robot: ArticulationCfg = STARLING2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=STARLING2_CFG.init_state.replace(pos=(0.0, 0.0, 0.1)),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=False,
    )




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
        randomize=False,
        max_action_delay_steps=0,
        motor_tau=0.02,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ProprioCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.03))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.05))

        rotation = ObsTerm(func=mdp.base_rotation_6d, noise=GaussianNoiseCfg(mean=0.0, std=0.02))


        x_error_yaw = ObsTerm(func=mdp.hover_x_error_yaw, noise=GaussianNoiseCfg(mean=0.0, std=0.01))
        y_error_yaw = ObsTerm(func=mdp.hover_y_error_yaw, noise=GaussianNoiseCfg(mean=0.0, std=0.01))
        height_error = ObsTerm(func=mdp.hover_height_error, noise=GaussianNoiseCfg(mean=0.0, std=0.05))
        yaw_error = ObsTerm(func=mdp.hover_yaw_error, noise=GaussianNoiseCfg(mean=0.0, std=0.02))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()


@configclass
class EventCfg:
    """Reset / startup events (incl."""

    set_hover_target_startup = EventTerm(
        func=mdp.reset_hover_target,
        mode="startup",
        params={"height_range": (2.0, 2.0)},
    )
    set_hover_target_reset = EventTerm(
        func=mdp.reset_hover_target,
        mode="reset",
        params={"height_range": (2.0, 2.0)},
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "z": (0.0, 0.1),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {
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
    record_episode_yaw_ref = EventTerm(
    func=mdp.record_episode_yaw_ref,
    mode="reset",
    params={"asset_cfg": SceneEntityCfg("robot")},
    )

    reset_airborne_flag = EventTerm(
        func=mdp.reset_airborne_flag,
        mode="reset",
    )



@configclass
class RewardsCfg:
    """Reward terms for the hover MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=0.1)

    height_asymmetric_rbf = RewTerm(
        func=mdp.hover_height_asymmetric_rbf,
        weight=10.0,
        params={"below_std": 0.3, "above_std": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )
    climb_when_below = RewTerm(
        func=mdp.hover_height_climb_when_below,
        weight=2.0,
        params={"height_tol": 0.3, "v_cap": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    vertical_damping_near_target = RewTerm(
        func=mdp.hover_vertical_damping_near_target,
        weight=-0.3,
        params={"near_tol": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )
    x_yaw_rbf = RewTerm(
        func=mdp.hold_x_yaw_rbf,
        weight=1.0,
        params={"std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    y_yaw_rbf = RewTerm(
        func=mdp.hold_y_yaw_rbf,
        weight=1.0,
        params={"std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    horizontal_velocity = RewTerm(
        func=mdp.horizontal_velocity_l2,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    yaw_tracking = RewTerm(
        func=mdp.episode_yaw_tracking,
        weight=1.5,
        params={"std": 0.3, "asset_cfg": SceneEntityCfg("robot")},
    )

    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.3)
    base_ang_vel = RewTerm(func=mdp.base_ang_vel_l2, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)

    crash_penalty = RewTerm(
        func=mdp.illegal_contact_takeoff_aware,
        weight=-20.0,
        params={
            "threshold": 1.0,
            "min_flight_height": 0.25,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    crash = DoneTerm(
        func=mdp.crashed_after_takeoff,
        params={
            "threshold": 1.0,
            "min_flight_height": 0.25,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    flipped = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},  # ~57 deg
    )


@configclass
class UavHoverEnvCfg(ManagerBasedRLEnvCfg):
    scene: UavHoverSceneCfg = UavHoverSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 10.0
        self.viewer.eye = (4.0, 0.0, 3.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
