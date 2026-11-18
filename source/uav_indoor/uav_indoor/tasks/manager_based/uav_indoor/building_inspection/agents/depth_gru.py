"""Compact disparity CNN followed by a GRU for local flight."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import RNN, HiddenState
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories
from tensordict import TensorDict


class DepthGruModel(MLPModel):
    """Encode one disparity frame before recurrent state integration."""

    is_recurrent = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (64,),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        cnn_channels: tuple[int, ...] | list[int] = (16, 32, 64, 64),
        cnn_kernel_sizes: tuple[int, ...] | list[int] = (4, 4, 4, 4),
        cnn_strides: tuple[int, ...] | list[int] = (2, 2, 2, 2),
        feature_dim: int = 192,
        rnn_hidden_dim: int = 192,
        rnn_num_layers: int = 1,
    ) -> None:
        self.feature_dim = feature_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        channels, height, width = self.disparity_shape
        if not (len(cnn_channels) == len(cnn_kernel_sizes) == len(cnn_strides)):
            raise ValueError("CNN channel, kernel, and stride lists must have equal length")
        encoder: list[nn.Module] = []
        input_channels = channels
        for output_channels, kernel_size, stride in zip(
            cnn_channels, cnn_kernel_sizes, cnn_strides, strict=True
        ):
            encoder.extend(
                (
                    nn.Conv2d(
                        input_channels,
                        int(output_channels),
                        kernel_size=int(kernel_size),
                        stride=int(stride),
                        padding=1,
                    ),
                    resolve_nn_activation(activation),
                )
            )
            height = (height + 2 - int(kernel_size)) // int(stride) + 1
            width = (width + 2 - int(kernel_size)) // int(stride) + 1
            if height <= 0 or width <= 0:
                raise ValueError("CNN configuration collapses the pooled image")
            input_channels = int(output_channels)
        self.image_spatial_shape = (height, width)
        encoder.append(nn.Flatten())
        self.disparity_encoder = nn.Sequential(*encoder)
        image_feature_count = input_channels * height * width
        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_count, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.vector_projection = nn.Sequential(
            nn.Linear(self.obs_dim, feature_dim),
            resolve_nn_activation(activation),
            nn.LayerNorm(feature_dim),
        )
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.rnn = RNN(feature_dim, rnn_hidden_dim, rnn_num_layers, "gru")

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        vector_groups: list[str] = []
        image_groups: list[str] = []
        reset_groups: list[str] = []
        vector_dim = 0
        for group in obs_groups[obs_set]:
            shape = obs[group].shape
            if group == "memory_reset":
                if len(shape) != 2 or shape[-1] != 1:
                    raise ValueError(f"memory_reset must have shape (N, 1), got {shape}")
                reset_groups.append(group)
            elif len(shape) == 4:
                image_groups.append(group)
            elif len(shape) == 2:
                vector_groups.append(group)
                vector_dim += shape[-1]
            else:
                raise ValueError(f"invalid observation shape for {group}: {shape}")

        if len(image_groups) != 1:
            raise ValueError(f"DepthGruModel requires one 2D observation group, got {image_groups}")
        if reset_groups != ["memory_reset"]:
            raise ValueError("DepthGruModel requires one memory_reset observation group")
        self.memory_reset_group = reset_groups[0]
        self.disparity_group = image_groups[0]
        self.disparity_shape = tuple(obs[self.disparity_group].shape[1:])
        return vector_groups, vector_dim

    def _get_latent_dim(self) -> int:
        return self.rnn_hidden_dim

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        vector = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        vector = self.vector_projection(self.obs_normalizer(vector))

        image = obs[self.disparity_group]
        image_flat = image.reshape(-1, *self.disparity_shape)
        image_feature = self.disparity_encoder(image_flat)
        image_feature = self.image_projection(image_feature)
        image_feature = image_feature.reshape(*image.shape[:-3], self.feature_dim)

        feature = self.feature_norm(vector + image_feature)
        reset = obs[self.memory_reset_group].squeeze(-1) > 0.5
        if masks is None:
            self.rnn.reset(reset)
            latent = self.rnn(feature).squeeze(0)
        else:
            if hidden_state is None:
                raise ValueError("hidden state is required for a recurrent policy update")

            outputs: list[torch.Tensor] = []
            state = hidden_state
            for step in range(feature.shape[0]):
                keep = (~reset[step]).to(dtype=feature.dtype).reshape(1, -1, 1)
                state = state * keep
                output, state = self.rnn.rnn(feature[step : step + 1], state)
                outputs.append(output)
            padded = torch.cat(outputs, dim=0)
            latent = unpad_trajectories(padded, masks).squeeze(0)
        self.latest_latent = latent
        return latent

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        return self.rnn.hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        self.rnn.detach_hidden_state(dones)


    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible actor with internal GRU state."""

        return _TorchDepthGruModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-compatible actor with explicit GRU state."""

        return _OnnxDepthGruModel(self, verbose)


class _DepthGruExportBase(nn.Module):
    """Copy the deterministic inference path without training-only wrappers."""

    def __init__(self, model: DepthGruModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.disparity_encoder = copy.deepcopy(model.disparity_encoder)
        self.image_projection = copy.deepcopy(model.image_projection)
        self.vector_projection = copy.deepcopy(model.vector_projection)
        self.feature_norm = copy.deepcopy(model.feature_norm)
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is None:
            self.deterministic_output = nn.Identity()
        else:
            self.deterministic_output = model.distribution.as_deterministic_output_module()

    def encode(self, vector: torch.Tensor, disparity: torch.Tensor) -> torch.Tensor:
        vector_feature = self.vector_projection(self.obs_normalizer(vector))
        image_feature = self.image_projection(self.disparity_encoder(disparity))
        return self.feature_norm(vector_feature + image_feature)


class _TorchDepthGruModel(_DepthGruExportBase):
    """TorchScript actor with recurrent state stored inside the module."""

    def __init__(self, model: DepthGruModel) -> None:
        super().__init__(model)
        self.register_buffer(
            "hidden_state",
            torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
        )

    def forward(self, vector: torch.Tensor, disparity: torch.Tensor) -> torch.Tensor:
        feature = self.encode(vector, disparity)
        latent, hidden = self.rnn(feature.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = hidden
        output = self.mlp(latent.squeeze(0))
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state.zero_()


class _OnnxDepthGruModel(_DepthGruExportBase):
    """ONNX actor with recurrent state supplied and returned by the caller."""

    is_recurrent = True

    def __init__(self, model: DepthGruModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim
        self.disparity_shape = model.disparity_shape

    def forward(
        self,
        vector: torch.Tensor,
        disparity: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.encode(vector, disparity)
        latent, hidden_out = self.rnn(feature.unsqueeze(0), hidden)
        output = self.mlp(latent.squeeze(0))
        return self.deterministic_output(output), hidden_out

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, self.input_size),
            torch.zeros(1, *self.disparity_shape),
            torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
        )

    @property
    def input_names(self) -> list[str]:
        return ["obs", "disparity", "h_in"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "h_out"]
