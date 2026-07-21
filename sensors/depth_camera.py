"""
depth_camera.py
══════════════════════════════════════════════════════════════════════════
Depth camera sensor: emits per-pixel depth (`DepthFrame`) and, optionally,
a derived `PointCloud`. Supports stereo-baseline parameters for
stereo-derived depth cameras as well as structured-light/ToF style
single-sensor depth cameras (baseline=0.0).
"""

from __future__ import annotations

from typing import Any, Optional

from base_sensor import Sensor
from sensor_types import (
    DepthFrame,
    PointCloud,
    PointCloudFormat,
    SensorType,
)

__all__ = ["DepthCamera"]


class DepthCamera(Sensor):
    """Depth-only or stereo-derived depth camera."""

    sensor_type = SensorType.DEPTH_CAMERA

    def __init__(
        self,
        name: str,
        *,
        resolution: tuple[int, int] = (1280, 720),
        fov_deg: float = 60.0,
        near_clip: float = 0.1,
        far_clip: float = 50.0,
        precision_m: float = 0.001,
        depth_encoding: str = "float32",
        stereo_baseline_m: float = 0.0,
        emit_point_cloud: bool = True,
        point_cloud_format: PointCloudFormat = PointCloudFormat.XYZ,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name,
            resolution=resolution,
            fov_deg=fov_deg,
            near_clip=near_clip,
            far_clip=far_clip,
            **kwargs,
        )
        self.precision_m = precision_m
        self.depth_encoding = depth_encoding
        self.stereo_baseline_m = stereo_baseline_m
        self.emit_point_cloud = emit_point_cloud
        self.point_cloud_format = point_cloud_format

    def _build_sample(self, raw_data: Any) -> DepthFrame:
        width, height = self.resolution
        depth_data = raw_data if isinstance(raw_data, list) else None

        point_cloud: Optional[PointCloud] = None
        if self.emit_point_cloud:
            point_cloud = PointCloud(
                **self._next_data_kwargs(),
                points=self._depth_to_points(depth_data, width, height),
                format=self.point_cloud_format,
            )

        return DepthFrame(
            **self._next_data_kwargs(),
            width=width,
            height=height,
            near=self.near_clip,
            far=self.far_clip,
            encoding=self.depth_encoding,
            depth_data=depth_data,
            point_cloud=point_cloud,
        )

    def _depth_to_points(self, depth_data: Optional[list[float]], width: int, height: int) -> list[tuple[float, ...]]:
        """Back-project a depth buffer into camera-space points using the
        pinhole model derived from `self.calibration.intrinsic`. Returns
        an empty list when no depth buffer is supplied (framework does
        not render; a backend adapter supplies `raw_data`)."""
        if not depth_data or self.calibration.intrinsic is None:
            return []
        intr = self.calibration.intrinsic
        cam_matrix = intr.camera_matrix(width, height)
        fx, cx = cam_matrix[0][0], cam_matrix[0][2]
        fy, cy = cam_matrix[1][1], cam_matrix[1][2]
        points: list[tuple[float, ...]] = []
        for idx, depth in enumerate(depth_data):
            if depth <= 0.0:
                continue
            px, py = idx % width, idx // width
            x = (px - cx) * depth / fx
            y = (py - cy) * depth / fy
            points.append((x, y, depth))
        return points

    def _validate_specific(self) -> list[str]:
        problems: list[str] = []
        if self.precision_m <= 0:
            problems.append(f"precision_m must be > 0 (got {self.precision_m}).")
        if self.stereo_baseline_m < 0:
            problems.append(f"stereo_baseline_m must be >= 0 (got {self.stereo_baseline_m}).")
        return problems

    def _serialize_specific(self) -> dict:
        return {
            "precision_m": self.precision_m,
            "depth_encoding": self.depth_encoding,
            "stereo_baseline_m": self.stereo_baseline_m,
            "emit_point_cloud": self.emit_point_cloud,
            "point_cloud_format": self.point_cloud_format.value,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DepthCamera":
        instance = super().deserialize(data)  # type: ignore[assignment]
        instance.precision_m = data.get("precision_m", 0.001)
        instance.depth_encoding = data.get("depth_encoding", "float32")
        instance.stereo_baseline_m = data.get("stereo_baseline_m", 0.0)
        instance.emit_point_cloud = data.get("emit_point_cloud", True)
        instance.point_cloud_format = PointCloudFormat(data.get("point_cloud_format", PointCloudFormat.XYZ.value))
        return instance
