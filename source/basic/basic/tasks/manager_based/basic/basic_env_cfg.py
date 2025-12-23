# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import (
    CommandTermCfg as CommandTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.sensors.imu import ImuCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from . import mdp  # use the standard locomotion mdp helpers from IsaacLab


##
# Terrain
##

TERRAIN_CONFIG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,        # meters
    num_cols=10,
    num_rows=10,
    curriculum=False,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.0,0.05),
            noise_step=0.01,
            downsampled_scale=0.5
        ),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.5),
            step_width=0.4,
        )
    },
)



@configclass
class MixedTerrainSceneCfg(InteractiveSceneCfg):
    """Go2 mixed-mesh terrain locomotion scene."""

    # ground / terrain
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path='/World/ground',
        terrain_generator=TERRAIN_CONFIG
    )
    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0,0,0.5)),
        attach_yaw_only=False,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=[1.0, 1.6],
        ),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    base_imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.01,
    )

    foot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_calf",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# Actions
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Joint position targets (delta from default pose), like Go2 rough env.
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,  # start conservative; you can tune later
    )


##
# Observations
##

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # proprioception & base state
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )

        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )

        last_actions = ObsTerm(
            func=mdp.last_action,
        )

        # commands (vx, vy, wz) in base frame
        commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # terrain / contact sensing (optional but helpful in sim)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
        )
        foot_contact_forces = ObsTerm(
            func=mdp.contact_forces,
            params={
                "sensor_cfg": SceneEntityCfg("foot_contacts"),
                'threshold': 50.0
            },
        )

        # IMU-derived measurements (can overlap with base_* terms; useful if you want teacher)
        imu_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            params={"asset_cfg": SceneEntityCfg("base_imu")},
            noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
        )
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("base_imu")},
            noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


##
# Commands (vx, vy, wz)
##

@configclass
class CommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=False,      # use angular velocity, not heading
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
        resampling_time_range=(1.0, 3.0),
        debug_vis=False,
    )

##
# Events
##

@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )

    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 4,      # number of discrete material buckets
        },
    )

    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",   # or "startup" if you want it only once
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "mass_distribution_params": (0.5, 1.5),  # example range
            "operation": "scale",                    # scale default mass by U[0.5, 1.5]
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    # optional pushes for robustness
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "force_range": (0.0, 150.0),   # [min, max] force magnitude in N
            "torque_range": (0.0, 0.0),    # or e.g. (0.0, 50.0) for random torques
        },
        interval_range_s=(3.0, 6.0),
    )

##
# Rewards
##

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # keep alive
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # tracking commands
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={
            "std": 0.05,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names="base")
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={
            "std": 0.05,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names="base")
        },
    )

    # posture & height
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "target_height": 0.33,
        },
    )

    # penalize high impact forces at the feet
    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-1e-3,
        params={
            "sensor_cfg": SceneEntityCfg("foot_contacts"),
            "threshold": 400.0,   # N; tune for your robot
        },
    )

    # penalize if none of the desired feet are in contact (discourage all-feet-in-air stance)
    no_desired_contact = RewTerm(
        func=mdp.desired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("foot_contacts"),
            "threshold": 50.0     # N; min force to consider as contact
        },
    )

    # penalize collisions of undesired links (e.g., thighs, body)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("foot_contacts"),
            "threshold": 5.0,
        },
    )

    # regularization
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-5e-4,
    )

    # termination penalty
    terminating = RewTerm(
        func=mdp.is_terminated,
        weight=-2.0,
    )


##
# Terminations
##

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 1) Time limit
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 2) Base tipped over (orientation too far from upright)
    base_fallen = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "limit_angle": 0.8,   # radians; ~45 deg, tune as needed
        },
    )

    # 3) Base too low (fallen / crouched excessively)
    base_height_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "minimum_height": 0.18,   # meters, tune for Go2
        },
    )


##
# Environment configuration
##

@configclass
class MixedTerrainGo2EnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL env for Unitree Go2 on mixed mesh terrain."""

    # scene
    scene: MixedTerrainSceneCfg = MixedTerrainSceneCfg(num_envs=32, env_spacing=4.0)

    # MDP pieces
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # viewer
        self.viewer.eye = (8.0, 0.0, 5.0)
        # sim settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
