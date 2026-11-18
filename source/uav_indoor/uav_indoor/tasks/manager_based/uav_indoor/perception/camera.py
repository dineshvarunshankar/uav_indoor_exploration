"""Calibrated camera geometry. ROS optical axes: +X right, +Y down, +Z forward."""

from __future__ import annotations

from dataclasses import dataclass

import torch

CALIBRATION_WIDTH = 1024
CALIBRATION_HEIGHT = 768
CALIBRATION_FX = 593.0477573442388
CALIBRATION_FY = 591.7125974187695
CALIBRATION_CX = 500.2332128799812
CALIBRATION_CY = 388.6282597746387
# Brown-Conrady coefficients: k1, k2, p1, p2, k3.
CALIBRATION_DISTORTION = (
    0.1388803712085015,
    -0.2150293276153785,
    0.0007812717807851107,
    0.002201948904745487,
    0.1215502080770114,
)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for one image size."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def aircraft(cls, width: int, height: int) -> CameraIntrinsics:
        """Scale the measured calibration to a render size."""

        scale_x = width / CALIBRATION_WIDTH
        scale_y = height / CALIBRATION_HEIGHT
        if abs(scale_x - scale_y) > 1e-6:
            raise ValueError(
                f"{width}x{height} is not the calibrated 4:3 shape: scaling would warp the image "
                f"({scale_x:.4f} across, {scale_y:.4f} down). Pick a size that keeps 4:3."
            )
        return cls(
            fx=CALIBRATION_FX * scale_x,
            fy=CALIBRATION_FY * scale_y,
            cx=CALIBRATION_CX * scale_x,
            cy=CALIBRATION_CY * scale_y,
            width=width,
            height=height,
        )

    def matrix(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """The 3x3 intrinsic matrix."""

        matrix = torch.zeros(3, 3, device=device)
        matrix[0, 0], matrix[1, 1] = self.fx, self.fy
        matrix[0, 2], matrix[1, 2] = self.cx, self.cy
        matrix[2, 2] = 1.0
        return matrix


def letterbox_points_to_source(
    points_xyn: torch.Tensor,
    source_width: int,
    source_height: int,
    letterbox_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map square detector coordinates back to the source camera image."""

    scale = min(letterbox_size / source_width, letterbox_size / source_height)
    resized_w = source_width * scale
    resized_h = source_height * scale
    pad_x = (letterbox_size - resized_w) * 0.5
    pad_y = (letterbox_size - resized_h) * 0.5

    pixel = points_xyn * float(letterbox_size)
    source_u = (pixel[..., 0] - pad_x) / scale
    source_v = (pixel[..., 1] - pad_y) / scale
    source = torch.stack((source_u / source_width, source_v / source_height), dim=-1)
    valid = (source[..., 0] >= 0.0) & (source[..., 0] <= 1.0) & (source[..., 1] >= 0.0) & (source[..., 1] <= 1.0)
    return source.clamp(0.0, 1.0), valid


def box_center_angular_error(
    box_xywhn: torch.Tensor, intrinsics: CameraIntrinsics
) -> torch.Tensor:
    """Return calibrated horizontal and vertical angles of box centers."""

    if box_xywhn.shape[-1] != 4:
        raise ValueError(f"box_xywhn must end with four values, got {tuple(box_xywhn.shape)}")
    u = box_xywhn[..., 0] * intrinsics.width
    v = box_xywhn[..., 1] * intrinsics.height
    return torch.stack(
        (
            torch.atan((u - intrinsics.cx) / intrinsics.fx),
            torch.atan((v - intrinsics.cy) / intrinsics.fy),
        ),
        dim=-1,
    )


def rays_from_normalised_points(points_xyn: torch.Tensor, intrinsics: CameraIntrinsics) -> torch.Tensor:
    """Unit rays in the optical frame for points given in [0, 1] image coordinates."""

    u = points_xyn[..., 0] * intrinsics.width
    v = points_xyn[..., 1] * intrinsics.height
    directions = torch.stack(
        (
            (u - intrinsics.cx) / intrinsics.fx,
            (v - intrinsics.cy) / intrinsics.fy,
            torch.ones_like(u),
        ),
        dim=-1,
    )
    return directions / directions.norm(dim=-1, keepdim=True)


def rotate_to_world(rays_optical: torch.Tensor, quat_w_ros: torch.Tensor) -> torch.Tensor:
    """Rotate optical-frame rays by the camera's world orientation (w, x, y, z)."""

    quat = quat_w_ros
    while quat.dim() < rays_optical.dim():
        quat = quat.unsqueeze(-2)
    w, x, y, z = quat[..., 0:1], quat[..., 1:2], quat[..., 2:3], quat[..., 3:4]
    t = 2.0 * torch.cross(torch.cat((x, y, z), dim=-1).expand_as(rays_optical), rays_optical, dim=-1)
    return rays_optical + w * t + torch.cross(
        torch.cat((x, y, z), dim=-1).expand_as(rays_optical), t, dim=-1
    )


def normalised_points_from_world(
    points_w: torch.Tensor,
    camera_position_w: torch.Tensor,
    quat_w_ros: torch.Tensor,
    intrinsics: CameraIntrinsics,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points into one camera without assuming a shared sensor pose."""

    quat_ros_w = quat_w_ros.clone()
    quat_ros_w[..., 1:] *= -1.0
    points_optical = rotate_to_world(points_w - camera_position_w, quat_ros_w)
    depth = points_optical[..., 2]
    safe_depth = torch.where(depth.abs() > 1.0e-6, depth, torch.ones_like(depth))
    u = intrinsics.fx * points_optical[..., 0] / safe_depth + intrinsics.cx
    v = intrinsics.fy * points_optical[..., 1] / safe_depth + intrinsics.cy
    points_xyn = torch.stack((u / intrinsics.width, v / intrinsics.height), dim=-1)
    visible = (
        (depth > 1.0e-6)
        & (points_xyn[..., 0] >= 0.0)
        & (points_xyn[..., 0] <= 1.0)
        & (points_xyn[..., 1] >= 0.0)
        & (points_xyn[..., 1] <= 1.0)
    )
    return points_xyn, visible


def project_depth_points_between_cameras(
    points_xyn: torch.Tensor,
    depth_m: torch.Tensor,
    point_valid: torch.Tensor,
    source_intrinsics: CameraIntrinsics,
    source_position_w: torch.Tensor,
    source_quat_w_ros: torch.Tensor,
    target_intrinsics: CameraIntrinsics,
    target_position_w: torch.Tensor,
    target_quat_w_ros: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project source-camera depth samples into a target camera image."""

    source_rays = rays_from_normalised_points(points_xyn, source_intrinsics)
    source_z = source_rays[..., 2].clamp_min(1.0e-6)
    source_points = source_rays * (depth_m / source_z).unsqueeze(-1)
    points_w = source_position_w[:, None, :] + rotate_to_world(
        source_points, source_quat_w_ros
    )

    target_quat_ros_w = target_quat_w_ros.clone()
    target_quat_ros_w[..., 1:] *= -1.0
    target_points = rotate_to_world(
        points_w - target_position_w[:, None, :], target_quat_ros_w
    )
    target_depth_m = target_points[..., 2]
    safe_depth = torch.where(
        target_depth_m.abs() > 1.0e-6,
        target_depth_m,
        torch.ones_like(target_depth_m),
    )
    u = target_intrinsics.fx * target_points[..., 0] / safe_depth + target_intrinsics.cx
    v = target_intrinsics.fy * target_points[..., 1] / safe_depth + target_intrinsics.cy
    target_xyn = torch.stack(
        (u / target_intrinsics.width, v / target_intrinsics.height), dim=-1
    )
    visible = (
        point_valid
        & (target_depth_m > 1.0e-6)
        & (target_xyn[..., 0] >= 0.0)
        & (target_xyn[..., 0] <= 1.0)
        & (target_xyn[..., 1] >= 0.0)
        & (target_xyn[..., 1] <= 1.0)
    )
    return target_xyn, target_depth_m, visible


def associate_projected_point(
    point_xyn: torch.Tensor,
    point_valid: torch.Tensor,
    boxes_xywhn: torch.Tensor,
    box_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match a projected ledger point to the closest live box that contains it."""

    delta = (boxes_xywhn[..., :2] - point_xyn.unsqueeze(-2)).abs()
    half_size = 0.5 * boxes_xywhn[..., 2:4].clamp_min(1.0e-6)
    contains = (delta <= half_size).all(dim=-1)
    candidates = contains & box_valid & point_valid.unsqueeze(-1)
    score = (delta / half_size).square().sum(dim=-1)
    score = torch.where(candidates, score, torch.full_like(score, torch.inf))
    index = score.argmin(dim=-1)
    matched = candidates.any(dim=-1)
    return index, matched


def intersect_plane(
    origin_w: torch.Tensor,
    rays_w: torch.Tensor,
    plane_normal_w: torch.Tensor,
    plane_offset_m: torch.Tensor,
    max_range_m: float = 120.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect world rays with a plane ``normal . x = offset``."""

    normal = plane_normal_w
    while normal.dim() < rays_w.dim():
        normal = normal.unsqueeze(-2)
    offset = plane_offset_m
    while offset.dim() < rays_w.dim() - 1:
        offset = offset.unsqueeze(-1)

    denominator = (rays_w * normal).sum(dim=-1)
    numerator = offset - (origin_w.unsqueeze(-2) * normal).sum(dim=-1)
    parallel = denominator.abs() < 1e-6
    distance = numerator / torch.where(parallel, torch.ones_like(denominator), denominator)
    valid = ~parallel & (distance > 0.0) & (distance < max_range_m)
    points = origin_w.unsqueeze(-2) + rays_w * distance.unsqueeze(-1)
    return points, valid
