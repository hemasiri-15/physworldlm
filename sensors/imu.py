"""
imu.py
══════════════════════════════════════════════════════════════════════════
IMU sensor: 3-axis accelerometer + gyroscope + optional magnetometer,
with independent bias/noise/thermal-drift per axis group, ADC saturation
modeling, covariance reporting, dual timestamp domains, and optional
gravity compensation on the accelerometer channel.

CHANGELOG vs previous version:
  - Saturation: accel/gyro/mag now clip to their configured range, and the
    reading reports which axes clipped (`*_saturated`).
  - Thermal drift: optional per-axis-group ThermalDriftModel adds a
    temperature-dependent bias offset on top of the static bias.
  - Magnetometer: `has_magnetometer=True` now actually produces a
    MagnetometerReading via `capture_magnetometer()`.
  - Timestamps: readings now carry hardware_timestamp / software_timestamp
    / clock_domain, in addition to the existing simulation `timestamp`.

REQUIRES (see sensor_types_additions.py):
  - sensor_types.ClockDomain
  - sensor_types.ThermalDriftModel
  - sensor_types.MagnetometerReading
  - sensor_types.IMUReading extended with hardware_timestamp,
    software_timestamp, clock_domain, accel_saturated, gyro_saturated
"""

from __future__ import annotations

import time
from typing import Any, Optional

from base_sensor import Sensor
from sensor_types import (
    ClockDomain,
    GaussianNoise,
    IMUReading,
    MagnetometerReading,
    NoiseModel,
    Quaternion,
    SensorType,
    ThermalDriftModel,
    Vec3,
)

__all__ = ["Imu"]

_GRAVITY_MS2 = 9.80665
_DEFAULT_REFERENCE_CELSIUS = 25.0


class Imu(Sensor):
    """Inertial measurement unit."""

    sensor_type = SensorType.IMU

    def __init__(
        self,
        name: str,
        *,
        accel_range_ms2: float = 156.9,   # ~16g
        gyro_range_rads: float = 34.9,    # ~2000 deg/s
        mag_range_ut: float = 4900.0,
        accel_bias: Optional[Vec3] = None,
        gyro_bias: Optional[Vec3] = None,
        mag_bias: Optional[Vec3] = None,
        accel_noise: Optional[NoiseModel] = None,
        gyro_noise: Optional[NoiseModel] = None,
        mag_noise: Optional[NoiseModel] = None,
        accel_covariance: tuple[float, ...] = (0.0,) * 9,
        gyro_covariance: tuple[float, ...] = (0.0,) * 9,
        orientation_covariance: tuple[float, ...] = (0.0,) * 9,
        mag_covariance: tuple[float, ...] = (0.0,) * 9,
        gravity_compensation: bool = True,
        has_magnetometer: bool = True,
        # --- thermal drift -------------------------------------------------
        accel_thermal_drift: Optional[ThermalDriftModel] = None,
        gyro_thermal_drift: Optional[ThermalDriftModel] = None,
        mag_thermal_drift: Optional[ThermalDriftModel] = None,
        # --- timestamps ------------------------------------------------------
        clock_domain: ClockDomain = ClockDomain.SIMULATION,
        hardware_clock_jitter_s: float = 0.0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("update_rate_hz", 200.0)
        super().__init__(name, **kwargs)
        self.accel_range_ms2 = accel_range_ms2
        self.gyro_range_rads = gyro_range_rads
        self.mag_range_ut = mag_range_ut
        self.accel_bias = accel_bias or Vec3()
        self.gyro_bias = gyro_bias or Vec3()
        self.mag_bias = mag_bias or Vec3()
        self.accel_noise = accel_noise or GaussianNoise(0.0, 0.0)
        self.gyro_noise = gyro_noise or GaussianNoise(0.0, 0.0)
        self.mag_noise = mag_noise or GaussianNoise(0.0, 0.0)
        self.accel_covariance = accel_covariance
        self.gyro_covariance = gyro_covariance
        self.orientation_covariance = orientation_covariance
        self.mag_covariance = mag_covariance
        self.gravity_compensation = gravity_compensation
        self.has_magnetometer = has_magnetometer

        # Thermal drift models (None => zero offset, i.e. no-op).
        self.accel_thermal_drift = accel_thermal_drift or ThermalDriftModel()
        self.gyro_thermal_drift = gyro_thermal_drift or ThermalDriftModel()
        self.mag_thermal_drift = mag_thermal_drift or ThermalDriftModel()

        # Timestamp domain config.
        self.clock_domain = clock_domain
        self.hardware_clock_jitter_s = hardware_clock_jitter_s

        # Ground-truth kinematic state, supplied per tick by the caller
        # via `set_true_state()`.
        self._true_linear_accel = Vec3()
        self._true_angular_vel = Vec3()
        self._true_orientation = Quaternion.identity()
        self._true_magnetic_field = Vec3(0.0, 0.0, 0.0)
        self._temperature_celsius = _DEFAULT_REFERENCE_CELSIUS

    # ------------------------------------------------------------------ #
    # State input
    # ------------------------------------------------------------------ #
    def set_true_state(
        self,
        linear_accel: Vec3,
        angular_vel: Vec3,
        orientation: Quaternion,
        *,
        magnetic_field: Optional[Vec3] = None,
        temperature_celsius: Optional[float] = None,
    ) -> None:
        """Feed ground-truth kinematic state (e.g. from the owning
        entity's `PhysicsState`) each tick before `capture()`.

        `magnetic_field` is the world-frame ambient field in microtesla
        (e.g. sourced from PW-WSF environment.schema.json's EM environment
        block). If omitted, the previously set value is retained.

        `temperature_celsius` feeds the thermal drift models. If omitted,
        the previously set value is retained (defaults to 25°C reference).
        """
        with self._lock:
            self._true_linear_accel = linear_accel
            self._true_angular_vel = angular_vel
            self._true_orientation = orientation
            if magnetic_field is not None:
                self._true_magnetic_field = magnetic_field
            if temperature_celsius is not None:
                self._temperature_celsius = temperature_celsius

    # ------------------------------------------------------------------ #
    # Shared per-axis-group pipeline: bias + thermal drift + noise + clip
    # ------------------------------------------------------------------ #
    def _apply_axis(
        self,
        v: Vec3,
        bias: Vec3,
        noise: NoiseModel,
        thermal_drift: ThermalDriftModel,
        range_limit: float,
    ) -> tuple[Vec3, tuple[bool, bool, bool]]:
        drift = thermal_drift.bias_offset(self._temperature_celsius)
        raw = Vec3(
            noise.apply(v.x + bias.x + drift.x),
            noise.apply(v.y + bias.y + drift.y),
            noise.apply(v.z + bias.z + drift.z),
        )
        return self._clip_to_range(raw, range_limit)

    @staticmethod
    def _clip_to_range(v: Vec3, limit: float) -> tuple[Vec3, tuple[bool, bool, bool]]:
        """Clamp each axis to [-limit, limit], reporting which axes clipped.

        Models ADC/output-register saturation: a real IMU driven beyond its
        configured full-scale range does not report the true value, it
        rails at the limit. `limit <= 0` disables clipping (treated as
        "no configured range" rather than "zero range").
        """
        if limit <= 0:
            return v, (False, False, False)

        def _clip(x: float) -> float:
            if x > limit:
                return limit
            if x < -limit:
                return -limit
            return x

        saturated = (abs(v.x) > limit, abs(v.y) > limit, abs(v.z) > limit)
        return Vec3(_clip(v.x), _clip(v.y), _clip(v.z)), saturated

    # ------------------------------------------------------------------ #
    # Timestamps
    # ------------------------------------------------------------------ #
    def _timestamp_kwargs(self, sim_timestamp: float) -> dict:
        """Derive hardware/software timestamps from the simulation clock.

        Simulation time is authoritative ground truth; hardware/software
        timestamps are modeled as sim time plus an optional fixed jitter,
        so downstream fusion code can be tested against clock skew without
        needing a real wall-clock in the sim loop.
        """
        hardware_ts = sim_timestamp + self.hardware_clock_jitter_s
        software_ts = sim_timestamp
        return {
            "hardware_timestamp": hardware_ts,
            "software_timestamp": software_ts,
            "clock_domain": self.clock_domain,
        }

    # ------------------------------------------------------------------ #
    # Capture: accel + gyro (+ orientation passthrough)
    # ------------------------------------------------------------------ #
    def _build_sample(self, raw_data: Any) -> IMUReading:
        accel = self._true_linear_accel
        if self.gravity_compensation:
            # Remove gravity expressed in the sensor's local frame, using
            # the current orientation estimate to rotate the world-frame
            # gravity vector into body frame.
            gravity_world = Vec3(0.0, -_GRAVITY_MS2, 0.0)
            gravity_body = self._true_orientation.rotate(gravity_world)
            accel = accel - gravity_body

        noisy_accel, accel_saturated = self._apply_axis(
            accel, self.accel_bias, self.accel_noise, self.accel_thermal_drift, self.accel_range_ms2
        )
        noisy_gyro, gyro_saturated = self._apply_axis(
            self._true_angular_vel, self.gyro_bias, self.gyro_noise, self.gyro_thermal_drift, self.gyro_range_rads
        )

        data_kwargs = self._next_data_kwargs()
        sim_timestamp = data_kwargs.get("timestamp", time.time())

        return IMUReading(
            **data_kwargs,
            **self._timestamp_kwargs(sim_timestamp),
            linear_acceleration=noisy_accel,
            angular_velocity=noisy_gyro,
            orientation=self._true_orientation,
            linear_acceleration_covariance=self.accel_covariance,
            angular_velocity_covariance=self.gyro_covariance,
            orientation_covariance=self.orientation_covariance,
            accel_saturated=accel_saturated,
            gyro_saturated=gyro_saturated,
        )

    # ------------------------------------------------------------------ #
    # Capture: magnetometer (separate reading type, separate call)
    # ------------------------------------------------------------------ #
    def capture_magnetometer(self) -> Optional[MagnetometerReading]:
        """Produce a magnetometer reading, or None if this unit has none.

        Kept as a distinct call (rather than folded into `capture()` /
        `_build_sample()`) so callers that only care about accel/gyro don't
        pay for or handle a field that may not exist on this hardware
        variant, and so magnetometer update rate can diverge from the
        accel/gyro rate later without restructuring the base Sensor loop.
        """
        if not self.has_magnetometer:
            return None

        with self._lock:
            world_field = self._true_magnetic_field
            orientation = self._true_orientation

        body_field = orientation.rotate(world_field)
        noisy_field, saturated = self._apply_axis(
            body_field, self.mag_bias, self.mag_noise, self.mag_thermal_drift, self.mag_range_ut
        )

        data_kwargs = self._next_data_kwargs()
        sim_timestamp = data_kwargs.get("timestamp", time.time())

        return MagnetometerReading(
            sensor_id=data_kwargs.get("sensor_id", self.name),
            timestamp=sim_timestamp,
            **self._timestamp_kwargs(sim_timestamp),
            magnetic_field=noisy_field,
            magnetic_field_covariance=self.mag_covariance,
            saturated=saturated,
        )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate_specific(self) -> list[str]:
        problems: list[str] = []
        if self.accel_range_ms2 <= 0:
            problems.append(f"accel_range_ms2 must be > 0 (got {self.accel_range_ms2}).")
        if self.gyro_range_rads <= 0:
            problems.append(f"gyro_range_rads must be > 0 (got {self.gyro_range_rads}).")
        if self.has_magnetometer and self.mag_range_ut <= 0:
            problems.append(f"mag_range_ut must be > 0 when has_magnetometer=True (got {self.mag_range_ut}).")
        if (
            len(self.accel_covariance) != 9
            or len(self.gyro_covariance) != 9
            or len(self.orientation_covariance) != 9
            or len(self.mag_covariance) != 9
        ):
            problems.append("covariance tuples must each have exactly 9 elements (row-major 3x3).")
        if self.hardware_clock_jitter_s < 0:
            problems.append(f"hardware_clock_jitter_s must be >= 0 (got {self.hardware_clock_jitter_s}).")
        return problems

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def _serialize_specific(self) -> dict:
        return {
            "accel_range_ms2": self.accel_range_ms2,
            "gyro_range_rads": self.gyro_range_rads,
            "mag_range_ut": self.mag_range_ut,
            "accel_bias": self.accel_bias.to_dict(),
            "gyro_bias": self.gyro_bias.to_dict(),
            "mag_bias": self.mag_bias.to_dict(),
            "accel_noise": self.accel_noise.to_dict(),
            "gyro_noise": self.gyro_noise.to_dict(),
            "mag_noise": self.mag_noise.to_dict(),
            "accel_covariance": list(self.accel_covariance),
            "gyro_covariance": list(self.gyro_covariance),
            "orientation_covariance": list(self.orientation_covariance),
            "mag_covariance": list(self.mag_covariance),
            "gravity_compensation": self.gravity_compensation,
            "has_magnetometer": self.has_magnetometer,
            "accel_thermal_drift": self.accel_thermal_drift.to_dict(),
            "gyro_thermal_drift": self.gyro_thermal_drift.to_dict(),
            "mag_thermal_drift": self.mag_thermal_drift.to_dict(),
            "clock_domain": self.clock_domain.value,
            "hardware_clock_jitter_s": self.hardware_clock_jitter_s,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Imu":
        instance = super().deserialize(data)  # type: ignore[assignment]
        instance.accel_range_ms2 = data.get("accel_range_ms2", 156.9)
        instance.gyro_range_rads = data.get("gyro_range_rads", 34.9)
        instance.mag_range_ut = data.get("mag_range_ut", 4900.0)
        instance.accel_bias = Vec3.from_dict(data.get("accel_bias", {}))
        instance.gyro_bias = Vec3.from_dict(data.get("gyro_bias", {}))
        instance.mag_bias = Vec3.from_dict(data.get("mag_bias", {}))
        instance.accel_covariance = tuple(data.get("accel_covariance", (0.0,) * 9))
        instance.gyro_covariance = tuple(data.get("gyro_covariance", (0.0,) * 9))
        instance.orientation_covariance = tuple(data.get("orientation_covariance", (0.0,) * 9))
        instance.mag_covariance = tuple(data.get("mag_covariance", (0.0,) * 9))
        instance.gravity_compensation = data.get("gravity_compensation", True)
        instance.has_magnetometer = data.get("has_magnetometer", True)
        instance.accel_thermal_drift = ThermalDriftModel.from_dict(data.get("accel_thermal_drift", {}))
        instance.gyro_thermal_drift = ThermalDriftModel.from_dict(data.get("gyro_thermal_drift", {}))
        instance.mag_thermal_drift = ThermalDriftModel.from_dict(data.get("mag_thermal_drift", {}))
        instance.clock_domain = ClockDomain(data.get("clock_domain", ClockDomain.SIMULATION.value))
        instance.hardware_clock_jitter_s = data.get("hardware_clock_jitter_s", 0.0)
        return instance
