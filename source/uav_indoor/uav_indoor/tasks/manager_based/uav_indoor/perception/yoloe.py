"""Zero-shot YOLOE detection for opening candidates and people."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from .prompts import DetectionRole, PromptVocabulary

YOLOE_MODEL_PATH = Path("/home/ubuntu/yoloe-26n-seg.pt")
PROMPT_ENCODER_PATH = Path("/home/ubuntu/mobileclip2_b.ts")
INFERENCE_IMAGE_SIZE = 416


@dataclass
class DetectionBatch:
    """Padded prompt detections for a camera batch, on the inference device."""

    boxes_xywhn: torch.Tensor  # (N, D, 4) normalized centre and size
    confidence: torch.Tensor  # (N, D)
    role_id: torch.Tensor  # (N, D) DetectionRole value
    valid: torch.Tensor  # (N, D)

    def valid_for_role(self, role: DetectionRole) -> torch.Tensor:
        """Return the validity mask of one mission role. Used as the phase filter."""

        return self.valid & (self.role_id == int(role))


def _use_local_prompt_encoder(encoder_path: Path) -> None:
    """Point Ultralytics at the local text encoder."""

    if not encoder_path.is_file():
        raise FileNotFoundError(f"YOLOE prompt encoder does not exist: {encoder_path}")
    from ultralytics.utils import SETTINGS

    SETTINGS["weights_dir"] = str(encoder_path.parent)


def _prefer_environment_filelock() -> None:
    """Provide the inference-safe symbol absent from Isaac Sim's old filelock."""

    try:
        import filelock
    except ImportError:
        return
    if not hasattr(filelock, "AsyncFileLock"):
        # Synchronous filelock fallback for inference.
        filelock.AsyncFileLock = filelock.FileLock


class YoloePromptDetector:
    """Ultralytics YOLOE runner with a fixed text-prompt vocabulary."""

    def __init__(
        self,
        model_path: str | Path = YOLOE_MODEL_PATH,
        vocabulary: PromptVocabulary | None = None,
        prompt_encoder_path: str | Path = PROMPT_ENCODER_PATH,
        device: str | int = 0,
        confidence: float = 0.25,
        image_size: int = INFERENCE_IMAGE_SIZE,
        quantize: int | str | None = 16,
        max_detections: int = 16,
    ) -> None:
        self.model_path = Path(model_path)
        self.vocabulary = vocabulary or PromptVocabulary.default()
        self.prompt_encoder_path = Path(prompt_encoder_path)
        self.device = device
        self.confidence = confidence
        self.image_size = image_size
        self.quantize = quantize
        self.max_detections = max_detections
        self.last_latency_s = 0.0
        self._model: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> Any:
        """Load the checkpoint and configure the prompt embeddings once."""

        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise FileNotFoundError(f"YOLOE model does not exist: {self.model_path}")
        try:
            _prefer_environment_filelock()
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for YOLOE inference") from exc
        _use_local_prompt_encoder(self.prompt_encoder_path)
        model = YOLOE(str(self.model_path))
        prompts = list(self.vocabulary.prompts)
        model.set_classes(prompts, model.get_text_pe(prompts))
        self._model = model
        return model

    def infer_batch(self, rgb: torch.Tensor) -> DetectionBatch:
        """Run batched inference on an NHWC uint8 device tensor. Masks stay unused."""

        model = self.load()
        start = perf_counter()
        images = rgb[..., :3].permute(0, 3, 1, 2).float().div_(255.0)
        results = model.predict(
            source=images,
            imgsz=self.image_size,
            conf=self.confidence,
            max_det=self.max_detections,
            quantize=self.quantize,
            device=self.device,
            verbose=False,
            stream=False,
        )
        self.last_latency_s = perf_counter() - start

        batch = len(results)
        device = images.device
        role_lookup = self.vocabulary.role_lookup(device)
        boxes = torch.zeros(batch, self.max_detections, 4, device=device)
        confidence = torch.zeros(batch, self.max_detections, device=device)
        role_id = torch.zeros(batch, self.max_detections, dtype=torch.long, device=device)
        valid = torch.zeros(batch, self.max_detections, dtype=torch.bool, device=device)

        for index, result in enumerate(results):
            count = min(len(result.boxes), self.max_detections)
            if count == 0:
                continue
            boxes[index, :count] = result.boxes.xywhn[:count]
            confidence[index, :count] = result.boxes.conf[:count]
            role_id[index, :count] = role_lookup[result.boxes.cls[:count].long()]
            valid[index, :count] = True

        return DetectionBatch(boxes, confidence, role_id, valid)
