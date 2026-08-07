"""
thermal.py
══════════════════════════════════════════════════════════════════════════
Thermal (infrared) camera sensor: emits a per-pixel temperature map
(`ThermalFrame`) bounded by a configured temperature range, with
emissivity, dynamic range, and false-color rendering hints.
"""

from __future__ import annotations

from typing import Any, Optional

from .base_sensor import Sensor
from .sensor_types import GaussianNoise, SensorType, ThermalFrame

__all__ = ["ThermalCamera"]


class ThermalCamera(Sensor):
    """Long-wave infrared thermal camera."""

    sensor_type = SensorType.THERMAL_CAMERA

    def __init__(
        self,
        name: str,
        *,
        resolution: tuple[int, int] = (640, 480),
        fov_deg: float = 45.0,
        near_clip: float = 0.1,
        far_clip: float = 500.0,
        temperature_range_k: tuple[float, float] = (233.15, 673.15),  # -40C .. 400C
        emissivity: float = 0.95,
        heat_map: bool = True,
        dynamic_range_db: float = 60.0,
        false_color: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, resolution=resolution, fov_deg=fov_deg, near_clip=near_clip, far_clip=far_clip, **kwargs)
        self.temperature_range_k = temperature_range_k
        self.emissivity = emissivity
        self.heat_map = heat_map
        self.dynamic_range_db = dynamic_range_db
        self.false_color = false_color

    def _build_sample(self, raw_data: Any) -> ThermalFrame:
        width, height = self.resolution
        temp_data: Optional[list[float]] = raw_data if isinstance(raw_data, list) else None

        min_temp, max_temp = self.temperature_range_k
        if temp_data:
            if self.noise_model is not None:
                temp_data = [
                    min(max_temp, max(min_temp, self.noise_model.apply(t))) for t in temp_data
                ]
            observed_min, observed_max = min(temp_data), max(temp_data)
        else:
            observed_min, observed_max = min_temp, max_temp

        return ThermalFrame(
            **self._next_data_kwargs(),
            width=width,
            height=height,
            temperature_data=temp_data,
            min_temp_k=observed_min,
            max_temp_k=observed_max,
            false_color=self.false_color,
        )

    def _validate_specific(self) -> list[str]:
        problems: list[str] = []
        lo, hi = self.temperature_range_k
        if hi <= lo:
            problems.append(f"temperature_range_k high ({hi}) must be > low ({lo}).")
        if not (0.0 <= self.emissivity <= 1.0):
            problems.append(f"emissivity must be in [0, 1] (got {self.emissivity}).")
        if self.dynamic_range_db <= 0:
            problems.append(f"dynamic_range_db must be > 0 (got {self.dynamic_range_db}).")
        return problems

    def _serialize_specific(self) -> dict:
        return {
            "temperature_range_k": list(self.temperature_range_k),
            "emissivity": self.emissivity,
            "heat_map": self.heat_map,
            "dynamic_range_db": self.dynamic_range_db,
            "false_color": self.false_color,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "ThermalCamera":
        instance = super().deserialize(data)  # type: ignore[assignment]
        instance.temperature_range_k = tuple(data.get("temperature_range_k", (233.15, 673.15)))
        instance.emissivity = data.get("emissivity", 0.95)
        instance.heat_map = data.get("heat_map", True)
        instance.dynamic_range_db = data.get("dynamic_range_db", 60.0)
        instance.false_color = data.get("false_color", True)
        return instance
