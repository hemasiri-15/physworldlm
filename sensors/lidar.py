"""
lidar.py
══════════════════════════════════════════════════════════════════════════
LiDAR sensor: rotating (mechanical, 360°) or solid-state (fixed FOV),
emitting a `PointCloud` per scan. Beam geometry, range, rotation speed,
return mode, and intensity are all configurable so the same class covers
everything from a 16-beam automotive puck to a 128-beam survey unit or a
narrow-FOV solid-state array.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

from base_sensor import Sensor
from sensor_types import LidarKind, PointCloud, PointCloudFormat, SensorType

__all__ = ["Lidar"]

ReturnMode = Literal["single", "dual", "triple"]


class Lidar(Sensor):
    """Spinning or solid-state LiDAR."""

    sensor_type = SensorType.LIDAR

    def __init__(
        self,
        name: str,
        *,
        lidar_kind: LidarKind = LidarKind.ROTATING,
        beam_count: int = 32,
        horizontal_resolution_deg: float = 0.2,
        vertical_resolution_deg: float = 2.0,
        horizontal_fov_deg: float = 360.0,
        vertical_fov_deg: float = 40.0,
        range_min_m: float = 0.5,
        range_max_m: float = 200.0,
        rotation_speed_hz: float = 10.0,
        returns: ReturnMode = "single",
        intensity: bool = True,
        point_cloud_format: PointCloudFormat = PointCloudFormat.XYZI,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("update_rate_hz", rotation_speed_hz)
        kwargs.setdefault("fov_deg", horizontal_fov_deg)
        kwargs.setdefault("near_clip", range_min_m)
        kwargs.setdefault("far_clip", range_max_m)
        super().__init__(name, **kwargs)
        self.lidar_kind = lidar_kind
        self.beam_count = beam_count
        self.horizontal_resolution_deg = horizontal_resolution_deg
        self.vertical_resolution_deg = vertical_resolution_deg
        self.horizontal_fov_deg = horizontal_fov_deg
        self.vertical_fov_deg = vertical_fov_deg
        self.range_min_m = range_min_m
        self.range_max_m = range_max_m
        self.rotation_speed_hz = rotation_speed_hz
        self.returns: ReturnMode = returns
        self.intensity = intensity
        self.point_cloud_format = point_cloud_format

    def points_per_scan(self) -> int:
        """Nominal point count for one full scan given the configured geometry."""
        if self.lidar_kind is LidarKind.ROTATING:
            azimuth_steps = max(1, int(round(self.horizontal_fov_deg / max(self.horizontal_resolution_deg, 1e-6))))
            return azimuth_steps * self.beam_count
        azimuth_steps = max(1, int(round(self.horizontal_fov_deg / max(self.horizontal_resolution_deg, 1e-6))))
        elevation_steps = max(1, int(round(self.vertical_fov_deg / max(self.vertical_resolution_deg, 1e-6))))
        return azimuth_steps * elevation_steps

    def _build_sample(self, raw_data: Any) -> PointCloud:
        points: list[tuple[float, ...]]
        if isinstance(raw_data, list):
            points = raw_data
            if self.noise_model is not None:
                points = [self._apply_range_noise(p) for p in points]
        else:
            points = []

        return PointCloud(
            **self._next_data_kwargs(),
            points=points,
            format=self.point_cloud_format,
        )

    def _apply_range_noise(self, point: tuple[float, ...]) -> tuple[float, ...]:
        """Apply the configured noise model along the point's range
        (radial) direction, preserving bearing -- the physically correct
        place for LiDAR ranging noise rather than perturbing x/y/z
        independently."""
        x, y, z = point[0], point[1], point[2]
        r = math.sqrt(x * x + y * y + z * z)
        if r == 0.0:
            return point
        noisy_r = self.noise_model.apply(r)  # type: ignore[union-attr]
        scale = noisy_r / r
        new_point = (x * scale, y * scale, z * scale) + tuple(point[3:])
        return new_point

    def _validate_specific(self) -> list[str]:
        problems: list[str] = []
        if self.beam_count <= 0:
            problems.append(f"beam_count must be > 0 (got {self.beam_count}).")
        if self.range_max_m <= self.range_min_m:
            problems.append(f"range_max_m ({self.range_max_m}) must be > range_min_m ({self.range_min_m}).")
        if self.rotation_speed_hz <= 0:
            problems.append(f"rotation_speed_hz must be > 0 (got {self.rotation_speed_hz}).")
        if self.horizontal_resolution_deg <= 0 or self.vertical_resolution_deg <= 0:
            problems.append("horizontal/vertical_resolution_deg must be > 0.")
        if self.returns not in ("single", "dual", "triple"):
            problems.append(f"returns must be one of single/dual/triple (got {self.returns!r}).")
        return problems

    def _serialize_specific(self) -> dict:
        return {
            "lidar_kind": self.lidar_kind.value,
            "beam_count": self.beam_count,
            "horizontal_resolution_deg": self.horizontal_resolution_deg,
            "vertical_resolution_deg": self.vertical_resolution_deg,
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "vertical_fov_deg": self.vertical_fov_deg,
            "range_min_m": self.range_min_m,
            "range_max_m": self.range_max_m,
            "rotation_speed_hz": self.rotation_speed_hz,
            "returns": self.returns,
            "intensity": self.intensity,
            "point_cloud_format": self.point_cloud_format.value,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Lidar":
        instance = super().deserialize(data)  # type: ignore[assignment]
        instance.lidar_kind = LidarKind(data.get("lidar_kind", LidarKind.ROTATING.value))
        instance.beam_count = data.get("beam_count", 32)
        instance.horizontal_resolution_deg = data.get("horizontal_resolution_deg", 0.2)
        instance.vertical_resolution_deg = data.get("vertical_resolution_deg", 2.0)
        instance.horizontal_fov_deg = data.get("horizontal_fov_deg", 360.0)
        instance.vertical_fov_deg = data.get("vertical_fov_deg", 40.0)
        instance.range_min_m = data.get("range_min_m", 0.5)
        instance.range_max_m = data.get("range_max_m", 200.0)
        instance.rotation_speed_hz = data.get("rotation_speed_hz", 10.0)
        instance.returns = data.get("returns", "single")
        instance.intensity = data.get("intensity", True)
        instance.point_cloud_format = PointCloudFormat(data.get("point_cloud_format", PointCloudFormat.XYZI.value))
        return instance
