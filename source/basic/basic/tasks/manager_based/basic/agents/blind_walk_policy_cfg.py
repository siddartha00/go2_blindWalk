from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go2BlindWalkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # rollout / training horizon
    num_steps_per_env = 24          # 24–32 is common for legged tasks
    max_iterations = 8000           # gives ~O(10^8) samples with 4k envs
    save_interval = 200
    experiment_name = "go2_blind_walk"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    # normalization
    empirical_normalization = True  # normalize obs using running stats

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,          # a bit more exploration than cartpole
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",        # keep as in template
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
