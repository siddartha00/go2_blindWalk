# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Basic-Nav-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.basic_nav_env_cfg:ObstacleNavEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:basic_nav_policy_cfg.yaml",
    },
)

gym.register(
    id="Template-Obstacle-Nav-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obstacle_nav_env_cfg:ObstacleNavEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:basic_nav_policy_cfg.yaml",
    },
)