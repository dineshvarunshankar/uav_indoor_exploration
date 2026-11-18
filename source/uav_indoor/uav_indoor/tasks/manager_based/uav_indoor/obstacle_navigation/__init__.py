"""Random-goal obstacle-navigation curriculum components."""

import gymnasium as gym


gym.register(
    id="Template-Uav-Obstacle-Navigation-v0",
    entry_point=f"{__name__}.env:ObstacleNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ObstacleNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": "uav_indoor.tasks.manager_based.uav_indoor.building_inspection.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
