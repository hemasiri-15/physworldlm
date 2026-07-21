"""
camera.py
══════════════════════════════════════════════════════════════════════════
RGB camera sensor.

Simulator-independent representation of a pinhole (or lens-distorted)
RGB camera: resolution, FOV, near/far planes, lens/distortion model,
exposure/gamma/HDR, motion blur, rolling shutter, and image encoding.
`capture()` builds a structurally valid `ImageFrame`; actual pixel
rendering is the responsibility of a backend adapter, which may pass
rendered bytes in via `capture(raw_data=...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from base_sensor import Sensor
from sensor_types import (
    CalibrationData,
    ImageEncoding,
    ImageFrame,
    IntrinsicCalibration,
    NoiseModel,
    SensorType,
    TimingModel,
    Transform6DoF,
)

__all__ = ["LensType", "Camera"]


class LensType(Enum):
    PINHOLE = "pinhole"
    FISHEYE = "fisheye"
    ORTHOGRAPHIC = "orthographic"


class Camera(Sensor):
    """RGB camera. `resolution` is `(width_px, height_px)`."""

    sensor_type = SensorType.CAMERA

    def __init__(
        self,
        name: str,
        *,
        resolution: tuple[int, int] = (1920, 1080),
        fov_deg: float = 60.0,
        near_clip: float = 0.1,
        far_clip: float = 1000.0,
        lens_type: LensType = LensType.PINHOLE,
        distortion_coeffs: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0),
        exposure: float = 1.0,
        gamma: float = 2.2,
        hdr: bool = False,
        motion_blur: bool = False,
        rolling_shutter: bool = False,
        image_encoding: ImageEncoding = ImageEncoding.RGB8,
        **kwargs: Any,
    ) -> None:
        calibration = kwargs.pop("calibration", None) or CalibrationData(
            intrinsic=IntrinsicCalibration(distortion_coeffs=distortion_coeffs)
        )
        super().__init__(
            name,
            resolution=resolution,
            fov_deg=fov_deg,
            near_clip=near_clip,
            far_clip=far_clip,
            calibration=calibration,
            **kwargs,
        )
        self.lens_type = lens_type
        self.exposure = exposure
        self.gamma = gamma
        self.hdr = hdr
        self.motion_blur = motion_blur
        self.rolling_shutter = rolling_shutter
        self.image_encoding = image_encoding

    def projection_matrix(self) -> tuple:
        width, height = self.resolution
        return self.calibration.projection_matrix(width, height, self.near_clip, self.far_clip)

    def _build_sample(self, raw_data: Any) -> ImageFrame:
        width, height = self.resolution
        channels = 4 if self.image_encoding in (ImageEncoding.RGBA8,) else (
            1 if self.image_encoding in (ImageEncoding.MONO8, ImageEncoding.MONO16) else 3
        )
        data = raw_data if isinstance(raw_data, (bytes, bytearray)) else None
        return ImageFrame(
            **self._next_data_kwargs(),
            width=width,
            height=height,
            channels=channels,
            encoding=self.image_encoding,
            data=data,
            exposure=self.exposure,
            gamma=self.gamma,
        )

    def _validate_specific(self) -> list[str]:
        problems: list[str] = []
        if self.exposure < 0:
            problems.append(f"exposure must be >= 0 (got {self.exposure}).")
        if self.gamma <= 0:
            problems.append(f"gamma must be > 0 (got {self.gamma}).")
        return problems

    def _serialize_specific(self) -> dict:
        return {
            "lens_type": self.lens_type.value,
            "exposure": self.exposure,
            "gamma": self.gamma,
            "hdr": self.hdr,
            "motion_blur": self.motion_blur,
            "rolling_shutter": self.rolling_shutter,
            "image_encoding": self.image_encoding.value,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Camera":
        instance = super().deserialize(data)  # type: ignore[assignment]
        instance.lens_type = LensType(data.get("lens_type", LensType.PINHOLE.value))
        instance.exposure = data.get("exposure", 1.0)
        instance.gamma = data.get("gamma", 2.2)
        instance.hdr = data.get("hdr", False)
        instance.motion_blur = data.get("motion_blur", False)
        instance.rolling_shutter = data.get("rolling_shutter", False)
        instance.image_encoding = ImageEncoding(data.get("image_encoding", ImageEncoding.RGB8.value))
        return instance
