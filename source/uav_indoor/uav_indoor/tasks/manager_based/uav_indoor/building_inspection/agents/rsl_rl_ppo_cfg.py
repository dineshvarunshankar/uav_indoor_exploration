"""PPO configurations for building inspection and its memory ablation."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlRNNModelCfg,
)

from .diagnostic_ppo import DiagnosticPpoAlgorithmCfg

_ACTOR_DISTRIBUTION = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6)


@configclass
class DepthGruModelCfg(RslRlMLPModelCfg):
    """Configuration for the project-local DepthNav-style actor."""

    class_name: str = (
        "uav_indoor.tasks.manager_based.uav_indoor.building_inspection.agents.depth_gru:DepthGruModel"
    )
    cnn_channels: list[int] = [16, 32, 64, 64]
    cnn_kernel_sizes: list[int] = [4, 4, 4, 4]
    cnn_strides: list[int] = [2, 2, 2, 2]
    feature_dim: int = 192
    rnn_hidden_dim: int = 192
    rnn_num_layers: int = 1


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """One-layer GRU policy for short perception and motion history."""

    num_steps_per_env = 256
    max_iterations = 3000
    save_interval = 100
    experiment_name = "uav_building_inspection"
    obs_groups = {
        "actor": ["policy", "disparity", "memory_reset"],
        "critic": ["critic"],
    }
    actor = DepthGruModelCfg(
        hidden_dims=[64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=_ACTOR_DISTRIBUTION,
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=64,
        rnn_num_layers=1,
    )
    algorithm = DiagnosticPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=3,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


@configclass
class FeedForwardPPORunnerCfg(PPORunnerCfg):
    """Ablation with the same PPO setup and heads, without recurrent memory."""

    experiment_name = "uav_building_inspection_ff"
    actor = RslRlCNNModelCfg(
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=_ACTOR_DISTRIBUTION,
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[16, 32, 64, 64],
            kernel_size=[4, 4, 4, 4],
            stride=[2, 2, 2, 2],
            padding="zeros",
            activation="elu",
            flatten=True,
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 64],
        activation="elu",
        obs_normalization=True,
    )
