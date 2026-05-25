# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the quadcopters"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
#from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##
STARLING2_USD = (
    "omniverse://airlab-nucleus.andrew.cmu.edu/Public/DTC/starling2_preliminary.usd"
)

STARLING2_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=STARLING2_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        
        pos=(48.52, -38.906, 0.855), 
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg( #add more actuators if required (for example, gimbal, gripper, etc.)
            joint_names_expr=[".*"], #all joints
            stiffness=0.0, # P gain for position control
            damping=0.0, # D gain for velocity control
        ),
    },
)
"""Configuration for the ModalAI Starling 2 quadcopter."""
