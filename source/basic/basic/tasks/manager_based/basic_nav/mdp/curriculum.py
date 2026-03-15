from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv, mdp
from isaaclab.assets import Articulation
from isaaclab.terrains import TerrainImporter
from _collections_abc import Sequence
import torch

def obstacle_terain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.5
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command: mdp.TerrainBasedPose2dCommand = env.command_manager.get_command("pose_command")
    robot_pos_w = asset.data.root_pos_w[env_ids, :2]
    goal_pos_w = command[env_ids, :2]
    distance_to_goal = torch.norm(goal_pos_w - robot_pos_w, dim=1)
    move_up = distance_to_goal < threshold
    is_terminal = env.termination_manager.time_outs[env_ids]
    move_down = is_terminal & (distance_to_goal > (threshold * 2.0))
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())