"""Prompted perception and short-term tracking for building-opening inspection."""

from .camera import (
    CameraIntrinsics,
    associate_projected_point,
    box_center_angular_error,
    intersect_plane,
    letterbox_points_to_source,
    normalised_points_from_world,
    project_depth_points_between_cameras,
    rays_from_normalised_points,
    rotate_to_world,
)
from .disparity import DisparityCfg, DisparityMonitor, ablate_disparity, inverse_depth_alignment
from .doorway import DOORWAY_MODEL_PATH, DoorwayDetector
from .midas import MIDAS_MODEL_PATH, MidasOnnxRunner, normalize_disparity
from .prompts import DetectionRole, PromptSpec, PromptVocabulary
from .structure import accumulate_rays, fit_vertical_plane, solve_accumulated
from .tof import TofCfg, TofPoints, TofSampler, TofTargetEvidence
from .tracking import TRACK_FEATURES, BatchedPromptBoxKalman, PromptBoxTrackers, TrackerCfg, select_active_box
from .yoloe import INFERENCE_IMAGE_SIZE, PROMPT_ENCODER_PATH, YOLOE_MODEL_PATH, DetectionBatch, YoloePromptDetector
