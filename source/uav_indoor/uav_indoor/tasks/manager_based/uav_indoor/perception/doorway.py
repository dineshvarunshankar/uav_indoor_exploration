"""Opening detector backend. Weights use the AGPL-3.0 license."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from .prompts import DetectionRole
from .yoloe import DetectionBatch, _prefer_environment_filelock

DOORWAY_MODEL_PATH = Path("/home/ubuntu/doorway_seg_yolo26n.pt")

# Class order of the published model: doorway, door, people, window, mirror.
_CLASS_ROLES = {
    0: DetectionRole.OPENING_CANDIDATE,
    1: DetectionRole.OPENING_CANDIDATE,
    2: DetectionRole.PERSON,
    3: DetectionRole.OPENING_CANDIDATE,
}


class DoorwayDetector:
    """Fixed-class detector with the same batch interface as the prompt detector."""

    def __init__(
        self,
        model_path: str | Path = DOORWAY_MODEL_PATH,
        device: str = "cuda",
        image_size: int = 384,
        confidence: float = 0.10,
        max_detections: int = 20,
    ) -> None:
        _prefer_environment_filelock()
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.max_detections = max_detections
        self.last_latency_s = 0.0

    def infer_batch(self, rgb: torch.Tensor) -> DetectionBatch:
        """Detect on ``(N, H, W, 3)`` uint8 images and pad to the batch layout."""

        # Convert RGB to BGR for Ultralytics.
        images = [frame.cpu().numpy()[..., ::-1] for frame in rgb[..., :3]]
        start = perf_counter()
        results = self.model.predict(
            source=images,
            imgsz=self.image_size,
            conf=self.confidence,
            max_det=self.max_detections,
            device=self.device,
            verbose=False,
            stream=False,
        )
        self.last_latency_s = perf_counter() - start

        device = rgb.device
        count = len(results)
        boxes = torch.zeros(count, self.max_detections, 4, device=device)
        confidence = torch.zeros(count, self.max_detections, device=device)
        role_id = torch.zeros(count, self.max_detections, dtype=torch.long, device=device)
        valid = torch.zeros(count, self.max_detections, dtype=torch.bool, device=device)

        height, width = rgb.shape[1], rgb.shape[2]
        for index, result in enumerate(results):
            if result.boxes is None or not len(result.boxes):
                continue
            slot = 0
            for detection in range(len(result.boxes)):
                role = _CLASS_ROLES.get(int(result.boxes.cls[detection]))
                if role is None or slot >= self.max_detections:
                    continue  # a mirror is not an opening
                x1, y1, x2, y2 = result.boxes.xyxy[detection].tolist()
                boxes[index, slot, 0] = ((x1 + x2) / 2) / width
                boxes[index, slot, 1] = ((y1 + y2) / 2) / height
                boxes[index, slot, 2] = (x2 - x1) / width
                boxes[index, slot, 3] = (y2 - y1) / height
                confidence[index, slot] = float(result.boxes.conf[detection])
                role_id[index, slot] = int(role)
                valid[index, slot] = True
                slot += 1
        return DetectionBatch(boxes_xywhn=boxes, confidence=confidence, role_id=role_id, valid=valid)
