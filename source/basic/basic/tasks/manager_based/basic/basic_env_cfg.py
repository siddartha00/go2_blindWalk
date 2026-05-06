# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UNoise

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from . import mdp  # use the standard locomotion mdp helpers from IsaacLab

##
# Terrain
##

TERRAIN_CONFIG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,        # meters
    num_cols=10,
    num_rows=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.05),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.0,0.25),
            noise_step=0.01,
            downsampled_scale=0.5
        ),
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.04, 0.1),
            step_width=0.2,
            platform_width=1.0,
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.04, 0.1),
            step_width=0.2,
            platform_width=4.0,
        ),
        "discrete_steps": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.0,
            platform_width=1.0,
            holes=False,
            grid_height_range=(0.01, 0.10),
            grid_width=0.45,
        ),
        "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            platform_width=1.0,
            slope_range=(0.04, 0.4)
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            platform_width=1.0,
            slope_range=(0.04, 0.4)
        )
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
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6,1.0)),
        ray_alignment="yaw",
        debug_vis=True,
        mesh_prim_paths=["/World/ground"]
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
# Actions
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Joint position targets (delta from default pose), like Go2 rough env.
    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True  # ~±0.5 rad deltas from standing[web:4][web:11]
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

        # observation terms (order preserved)
        # joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=UNoise(n_min=-0.05, n_max=0.05)
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=UNoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=UNoise(n_min=-0.2, n_max=0.2)
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=UNoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=UNoise(n_min=-1.5, n_max=1.5)
        )
        velocity_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        actions = ObsTerm(
            func=mdp.last_action,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg":SceneEntityCfg("height_scanner")},
            noise=UNoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0,1.0)
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


##
# Commands (vx, vy, wz)
##

@configclass
class CommandCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name='robot',
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi)
        )
    )

##
# Events
##

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

##
# Rewards
##

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- Reward to go at the given velocity
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=4.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- Penalties to maintain a smooth gait
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0) # Penalize vertical bouncing
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-4.5e-8) # Often needs to be higher than 1e-10 to matter
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0, # Returns positive value; positive weight = reward
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0, # Returns positive count of hits; negative weight = penalty
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*thigh",".*calf"]),
            "threshold": 1.0
        },
    )

    base_height = RewTerm(
        func=mdp.base_height,
        weight=1.6, # Returns NEGATIVE when below threshold; keep weight positive
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "threshold": 0.4,
        },
    )

    joint_devation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2, # Returns positive deviation; negative weight = penalty
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.5 # Penalty for not being flat
    )

    body_vel_alignment = RewTerm(
        func=mdp.base_heading_alignment,
        weight=0.1, # Returns NEGATIVE internally; keep weight positive to penalize
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # -- Penalty for moving out of joint limit bounds.
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=1.0 # Returns negative penalty; keep weight positive
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel
    )


##
# Terminations
##

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True
    )
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0}
    )


##
# Environment configuration
##

@configclass
class MixedTerrainGo2EnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL env for Unitree Go2 on mixed mesh terrain."""

    # Scene settings
    scene: MixedTerrainSceneCfg = MixedTerrainSceneCfg(num_envs=1024, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandCfg = CommandCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
