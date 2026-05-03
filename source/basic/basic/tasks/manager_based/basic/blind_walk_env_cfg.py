import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.sensors.imu import ImuCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UNoise

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from . import mdp
from .basic_env_cfg import TERRAIN_CONFIG, ActionsCfg, CommandCfg, CurriculumCfg, EventCfg, RewardsCfg, TerminationsCfg


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
            func=mdp.imu_projected_gravity,
            noise=UNoise(n_min=-0.05, n_max=0.05)
        )

        base_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            noise=UNoise(n_min=-0.1, n_max=0.1)
        )

        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
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

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True


    @configclass
    class CriticCfg(ObsGroup):

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
        )

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg":SceneEntityCfg("height_scanner")},
            noise=UNoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0,1.0)
        )

        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
        )

        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
        )

        velocity_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )

        actions = ObsTerm(
            func=mdp.last_action,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Go2BlindWalkEnvCfg(ManagerBasedRLEnvCfg):
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
