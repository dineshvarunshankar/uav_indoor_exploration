"""Quantized MiDaS depth inference on the GPU through ONNX Runtime."""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

MIDAS_MODEL_PATH = Path("/home/ubuntu/starling/AI_VOXL2/onnx/MiDaS/midas.onnx")
MIDAS_INPUT_SCALE = 0.00487531116232276
MIDAS_INPUT_ZERO_POINT = 24
MIDAS_OUTPUT_SCALE = 6.5142998695373535
MIDAS_OUTPUT_ZERO_POINT = 0
MIDAS_SIZE = 256


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian matching the deployed ``cv2.GaussianBlur(src, (0, 0), sigma)``."""

    radius = int(math.ceil(3.0 * sigma))
    offsets = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-offsets.square() / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    horizontal = F.pad(image, (radius, radius, 0, 0), mode="reflect")
    image = F.conv2d(horizontal, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
    vertical = F.pad(image, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(vertical, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)


def preprocess_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Apply the deployed antialiased resize and W8A8 quantization to an NHWC uint8 batch."""

    image = rgb[..., :3].permute(0, 3, 1, 2).float()
    scale = max(image.shape[-1] / MIDAS_SIZE, image.shape[-2] / MIDAS_SIZE)
    if scale > 1.5:
        image = _gaussian_blur(image, 0.5 * scale)
    image = F.interpolate(image, size=(MIDAS_SIZE, MIDAS_SIZE), mode="bilinear", align_corners=False)
    quantized = (image / 255.0 / MIDAS_INPUT_SCALE + MIDAS_INPUT_ZERO_POINT).round().clamp(0.0, 255.0)
    return quantized.permute(0, 2, 3, 1).to(torch.uint8).contiguous()


def normalize_disparity(disparity: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Robust per-frame normalization for the policy while retaining relative geometry."""

    flat = disparity.flatten(1)
    low = torch.quantile(flat, 0.02, dim=1)[:, None, None]
    high = torch.quantile(flat, 0.98, dim=1)[:, None, None]
    return ((disparity - low) / (high - low).clamp_min(eps)).clamp(0.0, 1.0)


class MidasOnnxRunner:
    """Batched inference that keeps the input and output in torch device memory."""

    def __init__(
        self,
        model_path: str | Path,
        num_envs: int,
        device: torch.device | str,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"MiDaS model does not exist: {path}")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"MiDaS inference requires a CUDA device, got {self.device}")

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), options, providers=["CUDAExecutionProvider"])

        model_input = self.session.get_inputs()[0]
        model_output = self.session.get_outputs()[0]
        if model_input.shape[1:] != [MIDAS_SIZE, MIDAS_SIZE, 3] or model_input.type != "tensor(uint8)":
            raise ValueError(f"Unexpected MiDaS input contract: shape={model_input.shape} type={model_input.type}")
        if model_output.shape[1:] != [MIDAS_SIZE, MIDAS_SIZE, 1] or model_output.type != "tensor(uint8)":
            raise ValueError(f"Unexpected MiDaS output contract: shape={model_output.shape} type={model_output.type}")

        device_index = self.device.index or 0
        self.input = torch.empty(num_envs, MIDAS_SIZE, MIDAS_SIZE, 3, dtype=torch.uint8, device=self.device)
        self.output = torch.empty(num_envs, MIDAS_SIZE, MIDAS_SIZE, 1, dtype=torch.uint8, device=self.device)
        self.binding = self.session.io_binding()
        self.binding.bind_input(
            model_input.name, "cuda", device_index, np.uint8, tuple(self.input.shape), self.input.data_ptr()
        )
        self.binding.bind_output(
            model_output.name, "cuda", device_index, np.uint8, tuple(self.output.shape), self.output.data_ptr()
        )
        self.last_latency_s = 0.0

    def infer_batch(self, rgb: torch.Tensor) -> torch.Tensor:
        """Return relative disparity as ``(N, 256, 256)`` on the model device."""

        start = perf_counter()
        self.input.copy_(preprocess_rgb(rgb))
        self.session.run_with_iobinding(self.binding)
        disparity = (self.output[..., 0].float() - MIDAS_OUTPUT_ZERO_POINT) * MIDAS_OUTPUT_SCALE
        torch.cuda.synchronize(self.device)
        self.last_latency_s = perf_counter() - start
        return disparity
