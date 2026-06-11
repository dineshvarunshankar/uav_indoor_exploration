# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
from gymnasium.envs.registration import WrapperSpec

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Uav-Hover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hover_env_cfg:UavHoverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
    additional_wrappers=(
        WrapperSpec(
            name="HoverEvalWrapper",
            entry_point=f"{__name__}.eval_wrapper:HoverEvalWrapper",
            kwargs={},
        ),
    ),
)
