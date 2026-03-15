# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import RayCaster, RayCasterCfg
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


def obstacle_range_scan_reward(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Penalize going near obstacles"""
    # command = env.command_manager.get_command(command_name)
    range_scanner: RayCaster = env.scene.sensors[sensor_cfg.name]
    range_scanner_pos = range_scanner.data.pos_w.unsqueeze(1)
    range_scan_hits = range_scanner.data.ray_hits_w
    range_scan_vectors = range_scan_hits - range_scanner_pos
    range_scan = torch.norm(range_scan_vectors, dim=-1)
    max_dist = range_scanner.cfg.max_distance
    range_scan = torch.where(range_scan < 0.1, torch.tensor(max_dist, device=env.device), range_scan)
    min_obstacle_distance, _ = torch.min(range_scan, dim=1)
    reward = torch.where(
        min_obstacle_distance < threshold,
        -10.0*(threshold - min_obstacle_distance)/threshold,
        torch.zeros_like(min_obstacle_distance)
    )
    return reward


def track_velocity_to_goal(env: ManagerBasedRLEnv, command_name: str, asset_name: str) -> torch.Tensor:
    """Reward velocity projected onto the vector pointing to the goal."""
    # 1. Get current command (target position) and robot state
    command = env.command_manager.get_command(command_name)
    robot = env.scene[asset_name]
    
    # 2. Get target position (World) and current position (World)
    # command.pos_command_w is [num_envs, 2]
    target_pos_w = command[:, :2] 
    current_pos_w = robot.data.root_pos_w[:, :2]
    
    # 3. Calculate direction vector to goal (Unit Vector)
    to_goal_vec = target_pos_w - current_pos_w
    to_goal_direction = torch.nn.functional.normalize(to_goal_vec, dim=-1)
    
    # 4. Get current linear velocity in World Frame
    lin_vel_w = robot.data.root_lin_vel_w[:, :2]
    
    # 5. Project velocity onto the goal direction (Dot Product)
    # This value is positive if moving toward goal, negative if moving away.
    velocity_toward_goal = torch.sum(lin_vel_w * to_goal_direction, dim=-1)
    
    # 6. Apply a small tanh or clipping to prevent "sprinting" 
    # and keep rewards bounded for stable training.
    return torch.clamp(velocity_toward_goal, min=-1.0, max=2.0)