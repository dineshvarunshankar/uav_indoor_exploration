"""Building-opening inspection task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Template-Uav-Building-Inspection-v0",
    entry_point=f"{__name__}.env:BuildingInspectionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:BuildingInspectionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Uav-Entry-Curriculum-v0",
    entry_point=f"{__name__}.env:BuildingInspectionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:EntryCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Uav-Building-Inspection-FF-v0",
    entry_point=f"{__name__}.env:BuildingInspectionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:BuildingInspectionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FeedForwardPPORunnerCfg",
    },
)
