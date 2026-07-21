"""
sensor_types_additions.py
══════════════════════════════════════════════════════════════════════════
MERGE TARGET: sensor_types.py

These are additive types/fields for the IMU upgrade (saturation, thermal
drift, magnetometer, timestamp domains). They follow the same conventions
as your existing GaussianNoise / Vec3 / IMUReading: dataclasses with
to_dict()/from_dict() for the serialize/deserialize round-trip, immutable
value objects, no surprise defaults that hide missing calibration data.

NOTE: I don't have your actual sensor_types.py source, so:
  - `IMUReading` below is shown as a FULL proposed replacement — diff it
    against your real one and only add the new fields (marked NEW).
  - `ClockDomain`, `ThermalDriftModel`, `MagnetometerReading` are new and
    can be pasted in as-is (adjust imports to match your file's actual
    import block).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Assumes Vec3, Quaternion already exist in this file, unchanged.


# ─────────────────────────────────────────────────────────────────────────
# Clock domains — needed once you have >1 sensor and want to reason about
# skew between hardware capture time and simulation/software time.
# ─────────────────────────────────────────────────────────────────────────
class ClockDomain(Enum):
    HARDWARE = "hardware"      # timestamp comes from a (simulated) device clock
    SOFTWARE = "software"      # timestamp comes from the host/driver receiving it
    SIMULATION = "simulation"  # timestamp is the authoritative sim-time (default)


# ─────────────────────────────────────────────────────────────────────────
# Thermal drift — linear bias-vs-temperature model per axis group.
# Deliberately simple (linear) for v1; swap in a polynomial or LUT-based
# model later without touching call sites, since Imu only ever calls
# .bias_offset(temperature_celsius).
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class ThermalDriftModel:
    """bias_offset = coefficient * (T - reference_celsius), per axis.

    `coefficient` units are sensor-output-units per degree Celsius
    (e.g. m/s^2 per °C for accel, rad/s per °C for gyro, µT per °C for mag).
    """

    coefficient: "Vec3" = field(default_factory=lambda: Vec3())
    reference_celsius: float = 25.0

    def bias_offset(self, temperature_celsius: float) -> "Vec3":
        delta = temperature_celsius - self.reference_celsius
        return Vec3(
            self.coefficient.x * delta,
            self.coefficient.y * delta,
            self.coefficient.z * delta,
        )

    def to_dict(self) -> dict:
        return {
            "coefficient": self.coefficient.to_dict(),
            "reference_celsius": self.reference_celsius,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ThermalDriftModel":
        return cls(
            coefficient=Vec3.from_dict(data.get("coefficient", {})),
            reference_celsius=data.get("reference_celsius", 25.0),
        )


# ─────────────────────────────────────────────────────────────────────────
# Magnetometer reading — mirrors the shape of IMUReading so downstream
# fusion code can treat it uniformly (timestamps, covariance, saturation).
# Kept as its own type rather than folded into IMUReading, since not every
# IMU variant carries a magnetometer and this avoids an Optional-field
# grab-bag on the primary reading type.
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class MagnetometerReading:
    sensor_id: str
    timestamp: float                    # simulation time (existing convention)
    hardware_timestamp: float           # NEW: device-clock capture time
    software_timestamp: float           # NEW: host-received time
    clock_domain: ClockDomain           # NEW: which of the above is authoritative

    magnetic_field: "Vec3"              # microtesla, body frame
    magnetic_field_covariance: tuple[float, ...]  # row-major 3x3

    saturated: tuple[bool, bool, bool] = (False, False, False)  # NEW

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp,
            "hardware_timestamp": self.hardware_timestamp,
            "software_timestamp": self.software_timestamp,
            "clock_domain": self.clock_domain.value,
            "magnetic_field": self.magnetic_field.to_dict(),
            "magnetic_field_covariance": list(self.magnetic_field_covariance),
            "saturated": list(self.saturated),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MagnetometerReading":
        return cls(
            sensor_id=data["sensor_id"],
            timestamp=data["timestamp"],
            hardware_timestamp=data.get("hardware_timestamp", data["timestamp"]),
            software_timestamp=data.get("software_timestamp", data["timestamp"]),
            clock_domain=ClockDomain(data.get("clock_domain", ClockDomain.SIMULATION.value)),
            magnetic_field=Vec3.from_dict(data["magnetic_field"]),
            magnetic_field_covariance=tuple(data.get("magnetic_field_covariance", (0.0,) * 9)),
            saturated=tuple(data.get("saturated", (False, False, False))),
        )


# ─────────────────────────────────────────────────────────────────────────
# IMUReading — proposed FULL definition with new fields marked.
# Diff against your real dataclass; only the NEW-marked lines are additions.
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class IMUReading:
    sensor_id: str
    timestamp: float                    # simulation time (existing)
    hardware_timestamp: float           # NEW
    software_timestamp: float           # NEW
    clock_domain: ClockDomain           # NEW

    linear_acceleration: "Vec3"
    angular_velocity: "Vec3"
    orientation: "Quaternion"

    linear_acceleration_covariance: tuple[float, ...]
    angular_velocity_covariance: tuple[float, ...]
    orientation_covariance: tuple[float, ...]

    accel_saturated: tuple[bool, bool, bool] = (False, False, False)  # NEW
    gyro_saturated: tuple[bool, bool, bool] = (False, False, False)   # NEW

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp,
            "hardware_timestamp": self.hardware_timestamp,
            "software_timestamp": self.software_timestamp,
            "clock_domain": self.clock_domain.value,
            "linear_acceleration": self.linear_acceleration.to_dict(),
            "angular_velocity": self.angular_velocity.to_dict(),
            "orientation": self.orientation.to_dict(),
            "linear_acceleration_covariance": list(self.linear_acceleration_covariance),
            "angular_velocity_covariance": list(self.angular_velocity_covariance),
            "orientation_covariance": list(self.orientation_covariance),
            "accel_saturated": list(self.accel_saturated),
            "gyro_saturated": list(self.gyro_saturated),
        }
