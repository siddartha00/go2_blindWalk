# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.sensors.ray_caster import patterns, RayCasterCfg
from isaaclab.sensors.imu import ImuCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort:skip


##
# Low-level model imports
##

from ..basic.blind_walk_env_cfg import Go2BlindWalkEnvCfg
from ..basic.basic_env_cfg import TERRAIN_CONFIG

LOW_LEVEL_ENV_CFG = Go2BlindWalkEnvCfg()

##
# Scene definition
##

INITIAL_CONFIG = terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
    num_objects=1,       
    height=2.0,         
    radius=0.6,         
    max_yx_angle=0.0,    
)

END_CONFIG = terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
    num_objects=15,
    height=2.0,
    radius=0.2,
    max_yx_angle=0.0,
)

TERRAIN_CONFIG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_cols=10,
    num_rows=10,
    curriculum=False,
    sub_terrains={
        # Cylinder obstacles: 2m high, 20cm to 1m wide (radius 0.1 to 0.5)
        "repeated_cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=0.5,
            abs_height_noise=(0.0,0.0),
            rel_height_noise=(1.0,1.0),
            platform_height=0.0,
            platform_width=1.5,
            object_params_start=INITIAL_CONFIG,
            object_params_end=END_CONFIG,
            flat_patch_sampling={
                "my_nav_targets": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=20,
                    patch_radius=0.6,
                    max_height_diff=0.05,
                    z_range=(-0.1, 0.1),
                )
            },
        ),
    },
)

@configclass
class MixedTerrainSceneCfg(InteractiveSceneCfg):
    """Go2 mixed-mesh terrain locomotion scene."""

    # ground plane
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path='/World/ground',
        terrain_generator=TERRAIN_CONFIG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False
    )

    # robot
    # robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6,1.0]),
        ray_alignment="yaw",
        debug_vis=True,
        mesh_prim_paths=["/World/ground"]
    )

    range_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0,0.0,0.0), rot=(0.0, 0.0, 0.0, 0.0)),
        pattern_cfg=patterns.LidarPatternCfg(
            channels=90,
            horizontal_fov_range=[0.0, 360.0],
            horizontal_res=5.0,
            vertical_fov_range=[-0.0, 60.0],
        ),
        debug_vis=True,
        mesh_prim_paths=['/World/ground'],
        ray_alignment="yaw",
        max_distance=3.0
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/imu",
        history_length=20,
        update_period=0.01,
        debug_vis=True,
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"logs/rsl_rl/go2_blind_walk/2026-01-10_22-31-18/exported/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity
        )
        pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "pose_command"}
        )
        lidar_scan = ObsTerm(
            func=mdp.lidar_range_normalized,
            params={"sensor_cfg": SceneEntityCfg("range_scanner")}
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(
        func = mdp.obstacle_terain_levels_vel
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "pose_command"},
    )
    obstacle_avoidance_reward = RewTerm(
        func=mdp.obstacle_range_scan_reward,
        weight=1.0,
        params={
            'sensor_cfg': SceneEntityCfg('range_scanner'),
            'threshold': 1.0
        }
    )
    velocity_towards_progress_reward = RewTerm(
        func=mdp.track_velocity_to_goal,
        weight=1.0,
        params={
            'command_name': 'pose_command',
            'asset_name': 'robot'
        }
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.TerrainBasedPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        # ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
        ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(heading=(-math.pi, math.pi))
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 10.0},
    )


##
# Environment configuration
##


@configclass
class ObstacleNavEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: MixedTerrainSceneCfg = MixedTerrainSceneCfg(num_envs=1024, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
