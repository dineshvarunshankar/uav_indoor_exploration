"""RSL-RL PPO runner for the casualty-inspection task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "uav_casualty_inspection"
    seed = 42
    device = "cuda:0"
    logger = "tensorboard"
    wandb_project = "UAV_Casualty_Inspection"
    obs_groups = {
        "actor": ["actor", "depth"],
        "critic": ["privileged"],
    }
    actor = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[4, 4, 4],
            stride=[2, 2, 2],
            padding="zeros",
            activation="elu",
            flatten=True,
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
