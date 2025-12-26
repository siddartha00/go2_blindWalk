# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def feet_air_time(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:,sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:,sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold)*first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:,:2], dim=1) > 0.1
    return reward


def base_height(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    height_scanner: RayCaster = env.scene.sensors[sensor_cfg.name]
    height_scan = torch.mean(height_scanner.data.ray_hits_w[...,2], dim=-1)
    base_z = env.scene["robot"].data.root_pos_w[:,2]
    relative_height = base_z - height_scan
    reward = torch.where(
        relative_height < threshold,
        -2.0*(threshold - relative_height),
        0.1
    )
    return reward