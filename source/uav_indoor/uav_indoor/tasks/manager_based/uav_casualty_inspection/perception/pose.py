"""YOLO26n-pose inference and deployable face-visibility features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

YOLO_POSE_MODEL_PATH = Path("/home/ubuntu/yolo26n-pose.pt")
YOLO_DETECT_MODEL_PATH = Path("/home/ubuntu/yolo26n.pt")
FACE_KEYPOINTS = (0, 1, 2, 3, 4)  # nose, left/right eye, left/right ear


def _prefer_environment_filelock() -> None:
    """Provide the inference-safe symbol absent from Isaac Sim's old filelock."""

    try:
        import filelock
    except ImportError:
        return
    if not hasattr(filelock, "AsyncFileLock"):
        # Ultralytics imports this for asynchronous dataset conversion. Local
        # checkpoint inference never enters that path, so the synchronous lock
        # is a compatible stand-in and avoids modifying Isaac Sim's environment.
        filelock.AsyncFileLock = filelock.FileLock


@dataclass
class PoseBatch:
    """Padded pose detections for a camera batch, on the inference device."""

    boxes_xywhn: torch.Tensor
    keypoints_xyn: torch.Tensor
    keypoint_confidence: torch.Tensor
    confidence: torch.Tensor
    valid: torch.Tensor

    @property
    def face_confidence(self) -> torch.Tensor:
        """Conservative confidence from nose/eyes/ears visibility."""

        face = self.keypoint_confidence[..., FACE_KEYPOINTS]
        nose = face[..., 0]
        eyes = face[..., 1:3].amax(dim=-1)
        ears = face[..., 3:5].amax(dim=-1)
        return torch.minimum(nose, torch.maximum(eyes, ears)) * self.valid

    def face_visible(self, threshold: float = 0.35) -> torch.Tensor:
        return (self.face_confidence >= threshold) & self.valid


class YoloPoseRunner:
    """Ultralytics runner that keeps the supplied pose checkpoint unchanged."""

    def __init__(
        self,
        model_path: str | Path = YOLO_POSE_MODEL_PATH,
        device: str | int = 0,
        confidence: float = 0.25,
        max_detections: int = 12,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"YOLO pose model does not exist: {path}")
        try:
            _prefer_environment_filelock()
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for YOLO26n-pose inference") from exc
        self.model = YOLO(str(path))
        self.device = device
        self.confidence = confidence
        self.max_detections = max_detections
        self.last_latency_s = 0.0

    def infer_batch(self, rgb: torch.Tensor) -> PoseBatch:
        """Run batched pose inference on an NHWC uint8 device tensor."""

        start = perf_counter()
        images = rgb[..., :3].permute(0, 3, 1, 2).float().div_(255.0)
        results = self.model.predict(
            source=images,
            imgsz=tuple(images.shape[-2:]),
            conf=self.confidence,
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
            stream=False,
        )

        batch = len(results)
        device = images.device
        boxes = torch.zeros(batch, self.max_detections, 4, device=device)
        keypoints = torch.zeros(batch, self.max_detections, 17, 2, device=device)
        keypoint_conf = torch.zeros(batch, self.max_detections, 17, device=device)
        confidence = torch.zeros(batch, self.max_detections, device=device)
        valid = torch.zeros(batch, self.max_detections, dtype=torch.bool, device=device)

        for index, result in enumerate(results):
            count = min(len(result.boxes), self.max_detections)
            if count == 0:
                continue
            boxes[index, :count] = result.boxes.xywhn[:count]
            confidence[index, :count] = result.boxes.conf[:count]
            keypoints[index, :count] = result.keypoints.xyn[:count]
            keypoint_conf[index, :count] = result.keypoints.conf[:count]
            valid[index, :count] = True

        self.last_latency_s = perf_counter() - start
        return PoseBatch(boxes, keypoints, keypoint_conf, confidence, valid)
