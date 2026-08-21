"""Model-in-loop perception cache used by the casualty-inspection task."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .kalman import BatchedBoxKalman
from .midas import MIDAS_MODEL_PATH, MidasOnnxRunner, normalize_disparity
from .pose import YOLO_POSE_MODEL_PATH, YoloPoseRunner


def timestamps_aligned(
    depth_timestamp_s: torch.Tensor,
    pose_timestamp_s: torch.Tensor,
    maximum_skew_s: float = 0.03,
) -> torch.Tensor:
    """Validate paired model outputs before exposing them to the policy."""

    return torch.abs(depth_timestamp_s - pose_timestamp_s) <= maximum_skew_s


@dataclass(frozen=True)
class InspectionPerceptionCfg:
    midas_model_path: str = str(MIDAS_MODEL_PATH)
    pose_model_path: str = str(YOLO_POSE_MODEL_PATH)
    pose_device: str = "0"
    max_tracks: int = 8
    depth_height: int = 64
    depth_width: int = 64
    depth_history: int = 2
    depth_period_s: float = 0.2
    pose_period_s: float = 1.0 / 3.0
    vertical_fov_deg: float = 66.0
    nominal_person_height_m: float = 1.70
    photometric_randomization: bool = True
    frame_drop_probability: float = 0.03


class InspectionPerception:
    """Device-resident model pipeline with explicit timestamp freshness."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        cfg: InspectionPerceptionCfg | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = cfg or InspectionPerceptionCfg()
        self.midas = MidasOnnxRunner(self.cfg.midas_model_path, num_envs, self.device)
        self.pose = YoloPoseRunner(
            self.cfg.pose_model_path,
            device=self.cfg.pose_device,
            max_detections=self.cfg.max_tracks,
        )
        self.tracker = BatchedBoxKalman(num_envs, self.cfg.max_tracks, self.device)
        self.depth_history = torch.zeros(
            num_envs,
            self.cfg.depth_history,
            self.cfg.depth_height,
            self.cfg.depth_width,
            device=self.device,
        )
        self.depth_age_s = torch.full((num_envs,), torch.inf, device=self.device)
        self.pose_age_s = torch.full((num_envs,), torch.inf, device=self.device)
        self.last_depth_latency_s = 0.0
        self.last_pose_latency_s = 0.0
        self._depth_elapsed_s = self.cfg.depth_period_s
        self._pose_elapsed_s = self.cfg.pose_period_s

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.depth_history[env_ids] = 0.0
        self.depth_age_s[env_ids] = torch.inf
        self.pose_age_s[env_ids] = torch.inf
        self.tracker.reset(env_ids)

    def _randomize_photometry(self, rgb: torch.Tensor) -> torch.Tensor:
        image = rgb.float() / 255.0
        gain = torch.empty(self.num_envs, 1, 1, 1, device=self.device).uniform_(0.65, 1.35)
        gamma = torch.empty(self.num_envs, 1, 1, 1, device=self.device).uniform_(0.80, 1.20)
        color = torch.empty(self.num_envs, 1, 1, 3, device=self.device).uniform_(0.90, 1.10)
        noise = torch.randn_like(image) * 0.01
        image = (image * gain * color).clamp(0.0, 1.0).pow(gamma).add(noise).clamp(0.0, 1.0)
        return (image * 255.0).to(torch.uint8)

    def tick(self, dt: float, rgb: torch.Tensor) -> bool:
        """Advance ages and run each model on its own deployed inference period.

        Both models run in every stage and at one rate. Running pose only from stage 1 left
        the 96 track inputs at zero for the whole of stage 0, so ``obs_normalization``
        learned zero mean and variance and the policy collapsed the moment real values
        arrived. A lower rate in stage 0 would shift the age and freshness inputs instead.
        """

        self.depth_age_s += dt
        self.pose_age_s += dt
        self.tracker.predict(dt)
        self._depth_elapsed_s += dt
        self._pose_elapsed_s += dt

        run_depth = self._depth_elapsed_s + 1.0e-9 >= self.cfg.depth_period_s
        run_pose = self._pose_elapsed_s + 1.0e-9 >= self.cfg.pose_period_s
        if not run_depth and not run_pose:
            return False

        frames = rgb[..., :3]
        if self.cfg.photometric_randomization:
            frames = self._randomize_photometry(frames)

        if run_depth:
            self._depth_elapsed_s %= self.cfg.depth_period_s
            accepted = torch.rand(self.num_envs, device=self.device) >= self.cfg.frame_drop_probability
            disparity = normalize_disparity(self.midas.infer_batch(frames))
            depth = F.interpolate(
                disparity.unsqueeze(1),
                size=(self.cfg.depth_height, self.cfg.depth_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            updated_history = torch.roll(self.depth_history, shifts=-1, dims=1)
            updated_history[:, -1] = depth
            self.depth_history[accepted] = updated_history[accepted]
            self.depth_age_s[accepted] = 0.0
            self.last_depth_latency_s = self.midas.last_latency_s

        if run_pose:
            self._pose_elapsed_s %= self.cfg.pose_period_s
            accepted = torch.rand(self.num_envs, device=self.device) >= self.cfg.frame_drop_probability
            poses = self.pose.infer_batch(frames)
            self.tracker.update(
                poses.boxes_xywhn,
                poses.confidence,
                poses.face_confidence,
                poses.valid & accepted[:, None],
            )
            self.pose_age_s[accepted] = 0.0
            self.last_pose_latency_s = self.pose.last_latency_s

        return True

    def estimated_track_positions_w(
        self,
        robot_position_w: torch.Tensor,
        robot_yaw_w: torch.Tensor,
    ) -> torch.Tensor:
        """Approximate world tracks from bbox geometry without simulator truth."""

        boxes = self.tracker.state[..., :4]
        cx = boxes[..., 0]
        cy = boxes[..., 1]
        vertical_fov = math.radians(self.cfg.vertical_fov_deg)
        range_m = self.track_range_m
        aspect = 4.0 / 3.0
        horizontal_fov = 2.0 * math.atan(math.tan(vertical_fov / 2.0) * aspect)
        bearing = robot_yaw_w[:, None] - (cx - 0.5) * horizontal_fov
        elevation = -(cy - 0.5) * vertical_fov
        horizontal_range = range_m * torch.cos(elevation)
        position = robot_position_w[:, None, :].expand(-1, self.cfg.max_tracks, -1).clone()
        position[..., 0] += horizontal_range * torch.cos(bearing)
        position[..., 1] += horizontal_range * torch.sin(bearing)
        position[..., 2] += range_m * torch.sin(elevation)
        return position

    @property
    def track_range_m(self) -> torch.Tensor:
        """Range to each track from the longest box side against a nominal person length.

        A person measures about 1.70 m along their longest axis whether they stand or lie
        down, so the longer box side is the stable one. Box height alone reads a lying
        casualty as small and puts them 3.8x too far away.
        """

        box_extent = self.tracker.state[..., 2:4].amax(dim=-1).clamp_min(0.02)
        focal_norm = 0.5 / math.tan(math.radians(self.cfg.vertical_fov_deg) / 2.0)
        return (self.cfg.nominal_person_height_m * focal_norm / box_extent).clamp(0.5, 25.0)

    @property
    def nearest_track_range_m(self) -> torch.Tensor:
        """Range to the closest valid track, or ``inf`` when nothing is tracked."""

        ranges = torch.where(self.tracker.valid, self.track_range_m, torch.inf)
        return ranges.min(dim=1).values

    @property
    def freshness(self) -> torch.Tensor:
        """Depth/pose freshness masks plus normalized age."""

        depth_period = self.cfg.depth_period_s
        pose_period = self.cfg.pose_period_s
        return torch.stack(
            [
                (self.depth_age_s <= 2.0 * depth_period).float(),
                (self.pose_age_s <= 2.0 * pose_period).float(),
                (self.depth_age_s / depth_period).clamp(0.0, 10.0),
                (self.pose_age_s / pose_period).clamp(0.0, 10.0),
            ],
            dim=-1,
        )
