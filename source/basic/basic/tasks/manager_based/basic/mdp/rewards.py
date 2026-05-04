# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    reward = torch.sum(torch.square(joint_pos - target), dim=1)
    return torch.nan_to_num(reward, nan=0.0)


def feet_air_time(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:,sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:,sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold)*first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:,:2], dim=1) > 0.1
    return torch.nan_to_num(reward, nan=0.0)


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
    return torch.nan_to_num(reward, nan=0.0)

def base_heading_alignment(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the difference between the robot's heading and its velocity direction."""
    # Extract the robot asset
    asset: Articulation = env.scene[asset_cfg.name]

    # Get the linear velocity in the world frame (X, Y components)
    vel_xy = asset.data.root_vel_w[:, :2]

    # Calculate the angle of the velocity vector: atan2(y, x)
    vel_heading = torch.atan2(vel_xy[:, 1], vel_xy[:, 0])

    # Get the robot's current yaw (heading) from the quaternion
    # Note: We use the projected gravity or standard quat-to-yaw math
    # Here we extract yaw from the root quaternion
    quat = asset.data.root_quat_w
    curr_heading = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2]**2 + quat[:, 3]**2)
    )

    # Compute the error and wrap it to (-pi, pi)
    heading_error = wrap_to_pi(curr_heading - vel_heading)

    # We want to minimize the squared error
    # Higher velocity makes the penalty more significant to prevent "drifting"
    reward = -torch.square(heading_error)

    # Only apply when the robot is actually moving (velocity > 0.1 m/s)
    # This prevents the robot from spinning in circles while standing still
    vel_norm = torch.norm(vel_xy, dim=1)
    reward *= (vel_norm > 0.1).float()

    return torch.nan_to_num(reward, nan=0.0)