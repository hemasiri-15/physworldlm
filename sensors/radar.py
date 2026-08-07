"""
radar.py
══════════════════════════════════════════════════════════════════════════
FMCW radar sensor: emits a set of tracked `RadarTarget`s (range, radial
velocity, azimuth/elevation, RCS) per scan, bounded by configured
range/velocity limits and a maximum track count.
"""

from __future__ import annotations

from typing import Any, Optional

from .base_sensor import Sensor
from .sensor_types import RadarKind, RadarTarget, RadarTargets, SensorType

__all__ = ["Radar"]


class Radar(Sensor):
    """FMCW radar."""

    sensor_type = SensorType.RADAR

    def __init__(
        self,
        name: str,
        *,
        radar_kind: RadarKind = RadarKind.FMCW,
        range_min_m: float = 0.2,
        range_max_m: float = 250.0,
        velocity_max_ms: float = 100.0,
        doppler_resolution_ms: float = 0.1,
        rcs_min_dbsm: float = -10.0,
        frequency_ghz: float = 77.0,
        bandwidth_mhz: float = 1000.0,
        beam_width_deg: float = 20.0,
        max_tracked_targets: int = 64,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("fov_deg", beam_width_deg)
        kwargs.setdefault("near_clip", range_min_m)
        kwargs.setdefault("far_clip", range_max_m)
        super().__init__(name, **kwargs)
        self.radar_kind = radar_kind
        self.range_min_m = range_min_m
        self.range_max_m = range_max_m
        self.velocity_max_ms = velocity_max_ms
        self.doppler_resolution_ms = doppler_resolution_ms
        self.rcs_min_dbsm = rcs_min_dbsm
        self.frequency_ghz = frequency_ghz
        self.bandwidth_mhz = bandwidth_mhz
        self.beam_width_deg = beam_width_deg
        self.max_tracked_targets = max_tracked_targets

    def range_resolution_m(self) -> float:
        """c / (2 * bandwidth) — standard FMCW range-resolution formula."""
        c = 299_792_458.0
        bandwidth_hz = self.bandwidth_mhz * 1e6
        if bandwidth_hz <= 0:
            return float("inf")
        return c / (2 * bandwidth_hz)

    def _build_sample(self, raw_data: Any) -> RadarTargets:
        targets: list[RadarTarget] = []
        if isinstance(raw_data, list):
            for entry in raw_data[: self.max_tracked_targets]:
                target = entry if isinstance(entry, RadarTarget) else RadarTarget(**entry)
                if self.noise_model is not None:
                    target = RadarTarget(
                        range_m=self.noise_model.apply(target.range_m),
                        velocity_ms=target.velocity_ms,
                        azimuth_rad=target.azimuth_rad,
                        elevation_rad=target.elevation_rad,
                        rcs_dbsm=target.rcs_dbsm,
                        track_id=target.track_id,
                    )
                if target.rcs_dbsm >= self.rcs_min_dbsm and self.range_min_m <= target.range_m <= self.range_max_m:
                    targets.append(target)

        return RadarTargets(**self._next_data_kwargs(), targets=targets)

    def _validate_specific(self) -> list[str]:
        problems: list[str] = []
        if self.range_max_m <= self.range_min_m:
            problems.append(f"range_max_m ({self.range_max_m}) must be > range_min_m ({self.range_min_m}).")
        if self.velocity_max_ms <= 0:
            problems.append(f"velocity_max_ms must be > 0 (got {self.velocity_max_ms}).")
        if self.bandwidth_mhz <= 0:
            problems.append(f"bandwidth_mhz must be > 0 (got {self.bandwidth_mhz}).")
        if self.frequency_ghz <= 0:
            problems.append(f"frequency_ghz must be > 0 (got {self.frequency_ghz}).")
        if self.max_tracked_targets <= 0:
            problems.append(f"max_tracked_targets must be > 0 (got {self.max_tracked_targets}).")
        return problems

    def _serialize_specific(self) -> dict:
        return {
            "radar_kind": self.radar_kind.value,
            "range_min_m": self.range_min_m,
            "range_max_m": self.range_max_m,
            "velocity_max_ms": self.velocity_max_ms,
            "doppler_resolution_ms": self.doppler_resolution_ms,
            "rcs_min_dbsm": self.rcs_min_dbsm,
            "frequency_ghz": self.frequency_ghz,
            "bandwidth_mhz": self.bandwidth_mhz,
            "beam_width_deg": self.beam_width_deg,
            "max_tracked_targets": self.max_tracked_targets,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Radar":
        instance = super().deserialize(data)  # type: ignore[assignment]
        instance.radar_kind = RadarKind(data.get("radar_kind", RadarKind.FMCW.value))
        instance.range_min_m = data.get("range_min_m", 0.2)
        instance.range_max_m = data.get("range_max_m", 250.0)
        instance.velocity_max_ms = data.get("velocity_max_ms", 100.0)
        instance.doppler_resolution_ms = data.get("doppler_resolution_ms", 0.1)
        instance.rcs_min_dbsm = data.get("rcs_min_dbsm", -10.0)
        instance.frequency_ghz = data.get("frequency_ghz", 77.0)
        instance.bandwidth_mhz = data.get("bandwidth_mhz", 1000.0)
        instance.beam_width_deg = data.get("beam_width_deg", 20.0)
        instance.max_tracked_targets = data.get("max_tracked_targets", 64)
        return instance
