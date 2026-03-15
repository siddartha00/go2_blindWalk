import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

def lidar_range_normalized(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reads LiDAR distances, normalizes by max_distance, and clips to [-1.0, 1.0]."""
    
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    max_dist = sensor.cfg.max_distance
    relative_vectors = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    distances = torch.norm(relative_vectors, dim=-1)
    distances = torch.where(distances < 0.01, torch.tensor(max_dist, device=env.device), distances)
    normalized_distances = distances / max_dist
    rescaled_obs = 2.0 * normalized_distances - 1.0
    return torch.clamp(rescaled_obs, min=-1.0, max=1.0)