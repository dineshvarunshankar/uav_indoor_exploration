"""PPO wrapper that exposes KL and bounds adaptive learning-rate changes."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.utils import unpad_trajectories

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg


def explained_variance(
    returns: torch.Tensor,
    predictions: torch.Tensor,
    epsilon: float = 1.0e-8,
) -> float:
    """Return value-function explained variance, or zero for constant returns."""

    if returns.shape != predictions.shape or returns.numel() == 0:
        raise ValueError("returns and predictions must have the same non-empty shape")
    if not torch.isfinite(returns).all() or not torch.isfinite(predictions).all():
        raise ValueError("returns and predictions must be finite")
    return_variance = torch.var(returns.float(), unbiased=False)
    if return_variance <= epsilon:
        return 0.0
    residual_variance = torch.var((returns - predictions).float(), unbiased=False)
    return float((1.0 - residual_variance / return_variance).item())


class DiagnosticPPO(PPO):
    """Use upstream PPO unchanged while recording KL and enforcing configured LR bounds."""

    def __init__(
        self,
        *args,
        minimum_learning_rate: float = 5.0e-5,
        maximum_learning_rate: float = 3.0e-4,
        stopping_risk_auxiliary: bool = False,
        stopping_risk_learning_rate: float = 1.0e-4,
        maximum_stopping_risk_coefficient: float = 1.0,
        stopping_risk_calibration_fraction: float = 0.10,
        **kwargs,
    ) -> None:
        self.minimum_learning_rate = minimum_learning_rate
        self.maximum_learning_rate = maximum_learning_rate
        self.stopping_risk_auxiliary = stopping_risk_auxiliary
        self.maximum_stopping_risk_coefficient = maximum_stopping_risk_coefficient
        self.stopping_risk_calibration_fraction = stopping_risk_calibration_fraction
        self.stopping_risk_coefficient: float | None = None
        self._configured_learning_rate = float(kwargs.get("learning_rate", 1.0e-3))
        super().__init__(*args, **kwargs)
        self.stopping_risk_head: nn.Linear | None = None
        self.stopping_risk_optimizer: torch.optim.Adam | None = None
        if self.stopping_risk_auxiliary:
            if not hasattr(self.actor, "rnn_hidden_dim"):
                raise ValueError("stopping-risk auxiliary loss requires a recurrent actor")
            self.stopping_risk_head = nn.Linear(self.actor.rnn_hidden_dim, 1).to(self.device)
            self.stopping_risk_optimizer = torch.optim.Adam(
                self.stopping_risk_head.parameters(), lr=stopping_risk_learning_rate
            )
        self._learning_rate_hook = self.optimizer.register_step_pre_hook(
            self._enforce_learning_rate_bounds
        )

    def _enforce_learning_rate_bounds(self, optimizer, args, kwargs):
        self.learning_rate = min(
            self.maximum_learning_rate, max(self.minimum_learning_rate, self.learning_rate)
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = self.learning_rate
        return args, kwargs

    def _update_with_stopping_risk(self) -> dict[str, float]:
        if self.rnd or self.symmetry or self.is_multi_gpu:
            raise ValueError("stopping-risk auxiliary loss does not support RND, symmetry, or multi-GPU")
        if self.stopping_risk_head is None or self.stopping_risk_optimizer is None:
            raise RuntimeError("stopping-risk auxiliary state is not initialized")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_stopping_risk_loss = 0.0
        generator = self.storage.recurrent_mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for batch in generator:
            if "stopping_risk" not in batch.observations.keys():
                raise ValueError("stopping-risk auxiliary loss requires a stopping_risk observation")
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1.0e-8
                    )

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, distribution_params
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    for parameter_group in self.optimizer.param_groups:
                        parameter_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            label = unpad_trajectories(
                batch.observations["stopping_risk"], batch.masks
            ).detach()
            prediction = torch.sigmoid(self.stopping_risk_head(self.actor.latest_latent))
            if prediction.shape != label.shape:
                raise ValueError(
                    f"stopping-risk prediction {prediction.shape} does not match label {label.shape}"
                )
            stopping_risk_loss = F.smooth_l1_loss(prediction, label)
            if self.stopping_risk_coefficient is None:
                numerator = self.stopping_risk_calibration_fraction * abs(
                    surrogate_loss.detach().item()
                )
                denominator = max(stopping_risk_loss.detach().item(), 1.0e-8)
                self.stopping_risk_coefficient = min(
                    self.maximum_stopping_risk_coefficient, numerator / denominator
                )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
                + self.stopping_risk_coefficient * stopping_risk_loss
            )
            self.optimizer.zero_grad()
            self.stopping_risk_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.stopping_risk_head.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.stopping_risk_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_stopping_risk_loss += stopping_risk_loss.item()

        updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value": mean_value_loss / updates,
            "surrogate": mean_surrogate_loss / updates,
            "entropy": mean_entropy / updates,
            "stopping_risk": mean_stopping_risk_loss / updates,
            "stopping_risk_coefficient": float(self.stopping_risk_coefficient),
        }

    def update(self) -> dict[str, float]:
        stored_steps = self.storage.step
        value_explained_variance = explained_variance(
            self.storage.returns[:stored_steps],
            self.storage.values[:stored_steps],
        )
        kl_samples: list[float] = []
        original: Callable = self.actor.get_kl_divergence

        def measured_kl(old_parameters, new_parameters):
            value = original(old_parameters, new_parameters)
            kl_samples.append(float(value.mean().detach().item()))
            return value

        self.actor.get_kl_divergence = measured_kl
        try:
            if self.stopping_risk_auxiliary:
                losses = self._update_with_stopping_risk()
            else:
                losses = super().update()
        finally:
            self.actor.get_kl_divergence = original

        self._enforce_learning_rate_bounds(self.optimizer, (), {})

        losses["kl"] = sum(kl_samples) / len(kl_samples) if kl_samples else 0.0
        losses["learning_rate_ratio"] = self.learning_rate / self._configured_learning_rate
        losses["explained_variance"] = value_explained_variance
        return losses


    def save(self) -> dict:
        saved = super().save()
        if self.stopping_risk_auxiliary:
            if self.stopping_risk_head is None or self.stopping_risk_optimizer is None:
                raise RuntimeError("stopping-risk auxiliary state is not initialized")
            saved["stopping_risk_head_state_dict"] = self.stopping_risk_head.state_dict()
            saved["stopping_risk_optimizer_state_dict"] = (
                self.stopping_risk_optimizer.state_dict()
            )
            saved["stopping_risk_coefficient"] = self.stopping_risk_coefficient
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        auxiliary_keys = (
            "stopping_risk_head_state_dict",
            "stopping_risk_optimizer_state_dict",
            "stopping_risk_coefficient",
        )
        present = [key for key in auxiliary_keys if key in loaded_dict]
        if present and len(present) != len(auxiliary_keys):
            raise ValueError(f"checkpoint has partial stopping-risk state: {present}")
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        self.last_checkpoint_migration: list[str] = []
        if not self.stopping_risk_auxiliary:
            return load_iteration
        if self.stopping_risk_head is None or self.stopping_risk_optimizer is None:
            raise RuntimeError("stopping-risk auxiliary state is not initialized")
        if not present:
            self.last_checkpoint_migration = list(auxiliary_keys)
            return load_iteration
        self.stopping_risk_head.load_state_dict(
            loaded_dict["stopping_risk_head_state_dict"], strict=True
        )
        self.stopping_risk_optimizer.load_state_dict(
            loaded_dict["stopping_risk_optimizer_state_dict"]
        )
        self.stopping_risk_coefficient = loaded_dict["stopping_risk_coefficient"]
        return load_iteration

@configclass
class DiagnosticPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Algorithm config fields consumed by :class:`DiagnosticPPO`."""

    class_name: str = (
        "uav_indoor.tasks.manager_based.uav_indoor.building_inspection.agents.diagnostic_ppo:DiagnosticPPO"
    )
    minimum_learning_rate: float = 5.0e-5
    maximum_learning_rate: float = 3.0e-4
    stopping_risk_auxiliary: bool = False
    stopping_risk_learning_rate: float = 1.0e-4
    maximum_stopping_risk_coefficient: float = 1.0
    stopping_risk_calibration_fraction: float = 0.10
