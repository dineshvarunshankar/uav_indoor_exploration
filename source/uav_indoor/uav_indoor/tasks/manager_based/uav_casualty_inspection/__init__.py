"""Gym registration for the casualty-inspection mission."""

import gymnasium as gym

from . import agents

gym.register(
    id="Template-Uav-Casualty-Inspection-v0",
    entry_point=f"{__name__}.inspection_env:CasualtyInspectionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inspection_env_cfg:CasualtyInspectionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
