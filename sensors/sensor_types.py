"""
sensor_types.py
══════════════════════════════════════════════════════════════════════════
Shared primitives for the PhysWorldLM Sensors Framework.

Scope
-----
Pure data + math. No simulator imports (no Omniverse, no ROS, no Isaac
Sim). Everything here is JSON-serializable and safe to construct on any
backend. This module owns:

    * Geometry:            Vec3, Quaternion, Transform6DoF
    * Coordinate frames:   CoordinateFrame enum + CoordinateTransformer
    * Noise models:        Gaussian / Uniform / Bias / RandomWalk / Drift
    * Timing models:       Latency / PacketLoss / FrameDrop / TimeDelay
    * Calibration:         IntrinsicCalibration, ExtrinsicCalibration
    * Typed sensor output: ImageFrame, DepthFrame, ThermalFrame,
                            PointCloud, RadarTargets, GPSReading,
                            IMUReading

`base_sensor.py` and every concrete sensor (`camera.py`, `lidar.py`, ...)
import from here rather than redefining these primitives.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Optional, Protocol, runtime_checkable

__all__ = [
    # geometry
    "Vec3", "Quaternion", "Transform6DoF",
    # frames
    "CoordinateFrame", "CoordinateTransformer",
    # noise
    "NoiseModel", "GaussianNoise", "UniformNoise", "BiasNoise",
    "RandomWalkNoise", "DriftNoise",
    # timing
    "TimingModel", "LatencyModel", "PacketLossModel", "FrameDropModel", "TimeDelayModel",
    # calibration
    "IntrinsicCalibration", "ExtrinsicCalibration", "CalibrationData",
    # enums
    "SensorType", "SensorStatus", "ImageEncoding", "PointCloudFormat",
    "LidarKind", "RadarKind",
    # data outputs
    "SensorData", "ImageFrame", "DepthFrame", "ThermalFrame", "PointCloud",
    "RadarTarget", "RadarTargets", "GPSReading", "IMUReading",
    # errors
    "SensorTypesError", "SerializationError",
]


# ════════════════════════════════════════════════════════════════════════
# Errors
# ════════════════════════════════════════════════════════════════════════

class SensorTypesError(Exception):
    """Base exception for `sensor_types` failures."""


class SerializationError(SensorTypesError):
    """Raised when a JSON round-trip fails validation."""


# ════════════════════════════════════════════════════════════════════════
# Geometry
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, d: dict) -> "Vec3":
        return cls(x=d.get("x", 0.0), y=d.get("y", 0.0), z=d.get("z", 0.0))

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self) -> "Vec3":
        m = self.magnitude()
        if m == 0.0:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / m, self.y / m, self.z / m)

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )


@dataclass
class Quaternion:
    """Hamilton convention, (w, x, y, z), unit quaternion by construction intent."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> dict:
        return {"w": self.w, "x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, d: dict) -> "Quaternion":
        return cls(w=d.get("w", 1.0), x=d.get("x", 0.0), y=d.get("y", 0.0), z=d.get("z", 0.0))

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> "Quaternion":
        """Build from intrinsic Euler angles (radians), ZYX (yaw-pitch-roll) order."""
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return cls(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
        )

    def to_euler(self) -> Vec3:
        """Return (roll, pitch, yaw) radians as a Vec3(x=roll, y=pitch, z=yaw)."""
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x ** 2 + self.y ** 2)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w * self.y - self.z * self.x)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y ** 2 + self.z ** 2)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return Vec3(x=roll, y=pitch, z=yaw)

    def normalized(self) -> "Quaternion":
        n = math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
        if n == 0.0:
            return Quaternion.identity()
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def multiply(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(
            w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def rotate(self, v: Vec3) -> Vec3:
        """Rotate a vector by this quaternion."""
        qv = Quaternion(0.0, v.x, v.y, v.z)
        conj = Quaternion(self.w, -self.x, -self.y, -self.z)
        r = self.multiply(qv).multiply(conj)
        return Vec3(r.x, r.y, r.z)


@dataclass
class Transform6DoF:
    """A rigid-body pose: translation + rotation. Used for mount offsets,
    sensor extrinsics, and body/world frame poses alike."""

    translation: Vec3 = field(default_factory=Vec3)
    rotation: Quaternion = field(default_factory=Quaternion.identity)

    def to_dict(self) -> dict:
        return {"translation": self.translation.to_dict(), "rotation": self.rotation.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "Transform6DoF":
        return cls(
            translation=Vec3.from_dict(d.get("translation", {})),
            rotation=Quaternion.from_dict(d.get("rotation", {})),
        )

    def compose(self, child: "Transform6DoF") -> "Transform6DoF":
        """Compose `self` (parent) with `child` (local offset) -> world-space transform.

        Used to combine a sensor's mount offset with its parent entity's
        pose, without either side needing to know about the other.
        """
        rotated_translation = self.rotation.rotate(child.translation)
        return Transform6DoF(
            translation=self.translation + rotated_translation,
            rotation=self.rotation.multiply(child.rotation).normalized(),
        )

    @classmethod
    def identity(cls) -> "Transform6DoF":
        return cls(Vec3(), Quaternion.identity())


# ════════════════════════════════════════════════════════════════════════
# Coordinate frames
# ════════════════════════════════════════════════════════════════════════

class CoordinateFrame(Enum):
    ENU = "enu"            # East-North-Up, local tangent plane
    NED = "ned"             # North-East-Down, local tangent plane
    ECEF = "ecef"           # Earth-Centered Earth-Fixed
    BODY = "body"           # Vehicle/platform body frame
    SENSOR = "sensor"       # Sensor-local frame
    WORLD = "world"         # Simulation world / scene frame


_WGS84_A = 6378137.0            # semi-major axis, meters
_WGS84_F = 1.0 / 298.257223563  # flattening
_WGS84_E2 = _WGS84_F * (2 - _WGS84_F)


class CoordinateTransformer:
    """Stateless conversions between the frames in `CoordinateFrame`.

    Geodetic <-> ECEF uses the WGS84 ellipsoid. ENU <-> NED is a fixed
    axis permutation. ECEF <-> ENU requires a local tangent-plane origin
    (lat0/lon0/alt0), supplied per call so this class stays stateless
    and safe to share across sensors/threads.
    """

    @staticmethod
    def enu_to_ned(v: Vec3) -> Vec3:
        return Vec3(x=v.y, y=v.x, z=-v.z)

    @staticmethod
    def ned_to_enu(v: Vec3) -> Vec3:
        return Vec3(x=v.y, y=v.x, z=-v.z)

    @staticmethod
    def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Vec3:
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        n = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sin_lat ** 2)
        x = (n + alt_m) * cos_lat * math.cos(lon)
        y = (n + alt_m) * cos_lat * math.sin(lon)
        z = (n * (1 - _WGS84_E2) + alt_m) * sin_lat
        return Vec3(x, y, z)

    @staticmethod
    def ecef_to_geodetic(ecef: Vec3) -> tuple[float, float, float]:
        """Bowring's method. Returns (lat_deg, lon_deg, alt_m)."""
        x, y, z = ecef.x, ecef.y, ecef.z
        lon = math.atan2(y, x)
        p = math.sqrt(x ** 2 + y ** 2)
        lat = math.atan2(z, p * (1 - _WGS84_E2))
        for _ in range(5):
            sin_lat = math.sin(lat)
            n = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sin_lat ** 2)
            alt = p / math.cos(lat) - n
            lat = math.atan2(z, p * (1 - _WGS84_E2 * n / (n + alt)))
        sin_lat = math.sin(lat)
        n = _WGS84_A / math.sqrt(1 - _WGS84_E2 * sin_lat ** 2)
        alt = p / math.cos(lat) - n
        return math.degrees(lat), math.degrees(lon), alt

    @staticmethod
    def ecef_to_enu(ecef: Vec3, origin_lat_deg: float, origin_lon_deg: float, origin_alt_m: float) -> Vec3:
        origin_ecef = CoordinateTransformer.geodetic_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_m)
        d = ecef - origin_ecef
        lat = math.radians(origin_lat_deg)
        lon = math.radians(origin_lon_deg)
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        sin_lon, cos_lon = math.sin(lon), math.cos(lon)
        east = -sin_lon * d.x + cos_lon * d.y
        north = -sin_lat * cos_lon * d.x - sin_lat * sin_lon * d.y + cos_lat * d.z
        up = cos_lat * cos_lon * d.x + cos_lat * sin_lon * d.y + sin_lat * d.z
        return Vec3(east, north, up)

    @staticmethod
    def enu_to_ecef(enu: Vec3, origin_lat_deg: float, origin_lon_deg: float, origin_alt_m: float) -> Vec3:
        origin_ecef = CoordinateTransformer.geodetic_to_ecef(origin_lat_deg, origin_lon_deg, origin_alt_m)
        lat = math.radians(origin_lat_deg)
        lon = math.radians(origin_lon_deg)
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        sin_lon, cos_lon = math.sin(lon), math.cos(lon)
        dx = -sin_lon * enu.x - sin_lat * cos_lon * enu.y + cos_lat * cos_lon * enu.z
        dy = cos_lon * enu.x - sin_lat * sin_lon * enu.y + cos_lat * sin_lon * enu.z
        dz = cos_lat * enu.y + sin_lat * enu.z
        return origin_ecef + Vec3(dx, dy, dz)


# ════════════════════════════════════════════════════════════════════════
# Noise models — independent, reusable, stateful where physically required
# ════════════════════════════════════════════════════════════════════════

class NoiseModel(ABC):
    """Base class for a scalar or per-axis noise process.

    Every model implements `apply(value)`. Stateful models (random walk,
    drift) mutate internal state on each call; stateless models
    (Gaussian, Uniform, Bias) do not. `reset()` clears any accumulated
    state so a sensor's `reset()` can propagate cleanly.
    """

    kind: str = "noise"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    @abstractmethod
    def apply(self, value: float) -> float:
        """Return `value` perturbed by this noise process."""

    def reset(self) -> None:
        """Clear accumulated state. No-op for stateless models."""
        return None

    def to_dict(self) -> dict:
        return {"kind": self.kind, **self._params()}

    def _params(self) -> dict:
        return {}


class GaussianNoise(NoiseModel):
    kind = "gaussian"

    def __init__(self, mean: float = 0.0, stddev: float = 0.0, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.mean = mean
        self.stddev = stddev

    def apply(self, value: float) -> float:
        return value + self._rng.gauss(self.mean, self.stddev)

    def _params(self) -> dict:
        return {"mean": self.mean, "stddev": self.stddev}


class UniformNoise(NoiseModel):
    kind = "uniform"

    def __init__(self, low: float = 0.0, high: float = 0.0, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.low = low
        self.high = high

    def apply(self, value: float) -> float:
        return value + self._rng.uniform(self.low, self.high)

    def _params(self) -> dict:
        return {"low": self.low, "high": self.high}


class BiasNoise(NoiseModel):
    """A fixed, constant offset (e.g. sensor calibration bias)."""

    kind = "bias"

    def __init__(self, bias: float = 0.0) -> None:
        super().__init__(None)
        self.bias = bias

    def apply(self, value: float) -> float:
        return value + self.bias

    def _params(self) -> dict:
        return {"bias": self.bias}


class RandomWalkNoise(NoiseModel):
    """Stateful: each call integrates a small Gaussian step onto a
    running offset (models slowly-varying sensor error, e.g. gyro
    random walk)."""

    kind = "random_walk"

    def __init__(self, step_stddev: float = 0.0, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.step_stddev = step_stddev
        self._offset = 0.0

    def apply(self, value: float) -> float:
        self._offset += self._rng.gauss(0.0, self.step_stddev)
        return value + self._offset

    def reset(self) -> None:
        self._offset = 0.0

    def _params(self) -> dict:
        return {"step_stddev": self.step_stddev, "current_offset": self._offset}


class DriftNoise(NoiseModel):
    """Stateful: monotonic linear drift accumulated per call at `rate`
    units/call (e.g. clock drift, thermal drift)."""

    kind = "drift"

    def __init__(self, rate: float = 0.0) -> None:
        super().__init__(None)
        self.rate = rate
        self._accumulated = 0.0

    def apply(self, value: float) -> float:
        self._accumulated += self.rate
        return value + self._accumulated

    def reset(self) -> None:
        self._accumulated = 0.0

    def _params(self) -> dict:
        return {"rate": self.rate, "accumulated": self._accumulated}


# ════════════════════════════════════════════════════════════════════════
# Timing / channel models — latency, packet loss, dropped frames, delay
# ════════════════════════════════════════════════════════════════════════

class TimingModel(ABC):
    """Base class for a channel-timing effect applied to captures."""

    kind: str = "timing"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    @abstractmethod
    def next_delay_s(self) -> float:
        """Additional delay (seconds) to apply to the next capture."""

    def should_drop(self) -> bool:
        """Whether the next capture should be dropped entirely. Default: never."""
        return False

    def to_dict(self) -> dict:
        return {"kind": self.kind, **self._params()}

    def _params(self) -> dict:
        return {}


class LatencyModel(TimingModel):
    kind = "latency"

    def __init__(self, mean_s: float = 0.0, jitter_s: float = 0.0, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.mean_s = mean_s
        self.jitter_s = jitter_s

    def next_delay_s(self) -> float:
        return max(0.0, self.mean_s + self._rng.uniform(-self.jitter_s, self.jitter_s))

    def _params(self) -> dict:
        return {"mean_s": self.mean_s, "jitter_s": self.jitter_s}


class PacketLossModel(TimingModel):
    kind = "packet_loss"

    def __init__(self, loss_probability: float = 0.0, seed: Optional[int] = None) -> None:
        super().__init__(seed)
        self.loss_probability = max(0.0, min(1.0, loss_probability))

    def next_delay_s(self) -> float:
        return 0.0

    def should_drop(self) -> bool:
        return self._rng.random() < self.loss_probability

    def _params(self) -> dict:
        return {"loss_probability": self.loss_probability}


class FrameDropModel(TimingModel):
    """Deterministic-cadence frame dropping (drop every Nth frame)."""

    kind = "frame_drop"

    def __init__(self, drop_every_n: int = 0) -> None:
        super().__init__(None)
        self.drop_every_n = drop_every_n
        self._counter = 0

    def next_delay_s(self) -> float:
        return 0.0

    def should_drop(self) -> bool:
        if self.drop_every_n <= 0:
            return False
        self._counter += 1
        if self._counter >= self.drop_every_n:
            self._counter = 0
            return True
        return False

    def _params(self) -> dict:
        return {"drop_every_n": self.drop_every_n}


class TimeDelayModel(TimingModel):
    """Fixed, constant delay applied to every capture."""

    kind = "time_delay"

    def __init__(self, delay_s: float = 0.0) -> None:
        super().__init__(None)
        self.delay_s = delay_s

    def next_delay_s(self) -> float:
        return self.delay_s

    def _params(self) -> dict:
        return {"delay_s": self.delay_s}


# ════════════════════════════════════════════════════════════════════════
# Calibration
# ════════════════════════════════════════════════════════════════════════

@dataclass
class IntrinsicCalibration:
    """Pinhole-camera-style intrinsics. Applicable to Camera/DepthCamera/
    ThermalCamera; other sensors leave this at defaults / unused."""

    focal_length_x: float = 1.0
    focal_length_y: float = 1.0
    principal_point_x: float = 0.5
    principal_point_y: float = 0.5
    skew: float = 0.0
    distortion_coeffs: tuple[float, ...] = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0, 0.0))

    def camera_matrix(self, width_px: int, height_px: int) -> tuple[tuple[float, float, float], ...]:
        fx = self.focal_length_x * width_px
        fy = self.focal_length_y * height_px
        cx = self.principal_point_x * width_px
        cy = self.principal_point_y * height_px
        return (
            (fx, self.skew, cx),
            (0.0, fy, cy),
            (0.0, 0.0, 1.0),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IntrinsicCalibration":
        d = dict(d)
        if "distortion_coeffs" in d:
            d["distortion_coeffs"] = tuple(d["distortion_coeffs"])
        return cls(**d)


@dataclass
class ExtrinsicCalibration:
    """Sensor-to-parent (or sensor-to-body) rigid transform used for
    calibration purposes, distinct from the live `mount_transform` used
    for real-time mounting -- this is the calibrated/measured value,
    which may differ slightly from the nominal mount."""

    rotation: Quaternion = field(default_factory=Quaternion.identity)
    translation: Vec3 = field(default_factory=Vec3)

    def to_transform(self) -> Transform6DoF:
        return Transform6DoF(translation=self.translation, rotation=self.rotation)

    def to_dict(self) -> dict:
        return {"rotation": self.rotation.to_dict(), "translation": self.translation.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "ExtrinsicCalibration":
        return cls(
            rotation=Quaternion.from_dict(d.get("rotation", {})),
            translation=Vec3.from_dict(d.get("translation", {})),
        )


@dataclass
class CalibrationData:
    intrinsic: Optional[IntrinsicCalibration] = None
    extrinsic: Optional[ExtrinsicCalibration] = None
    calibrated: bool = False
    calibration_error_rms: float = 0.0

    def projection_matrix(self, width_px: int, height_px: int, near: float, far: float) -> tuple:
        """OpenGL-style perspective projection matrix derived from intrinsics."""
        if self.intrinsic is None:
            raise SensorTypesError("Cannot compute projection_matrix without intrinsic calibration.")
        cm = self.intrinsic.camera_matrix(width_px, height_px)
        fx, fy = cm[0][0], cm[1][1]
        cx, cy = cm[0][2], cm[1][2]
        a = 2 * fx / width_px
        b = 2 * fy / height_px
        c = (2 * cx / width_px) - 1
        d = (2 * cy / height_px) - 1
        e = -(far + near) / (far - near)
        f = -(2 * far * near) / (far - near)
        return (
            (a, 0.0, c, 0.0),
            (0.0, b, d, 0.0),
            (0.0, 0.0, e, f),
            (0.0, 0.0, -1.0, 0.0),
        )

    def to_dict(self) -> dict:
        return {
            "intrinsic": self.intrinsic.to_dict() if self.intrinsic else None,
            "extrinsic": self.extrinsic.to_dict() if self.extrinsic else None,
            "calibrated": self.calibrated,
            "calibration_error_rms": self.calibration_error_rms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationData":
        return cls(
            intrinsic=IntrinsicCalibration.from_dict(d["intrinsic"]) if d.get("intrinsic") else None,
            extrinsic=ExtrinsicCalibration.from_dict(d["extrinsic"]) if d.get("extrinsic") else None,
            calibrated=d.get("calibrated", False),
            calibration_error_rms=d.get("calibration_error_rms", 0.0),
        )


# ════════════════════════════════════════════════════════════════════════
# Enums shared across sensor classes
# ════════════════════════════════════════════════════════════════════════

class SensorType(Enum):
    CAMERA = "camera"
    DEPTH_CAMERA = "depth_camera"
    THERMAL_CAMERA = "thermal_camera"
    LIDAR = "lidar"
    RADAR = "radar"
    GPS = "gps"
    IMU = "imu"


class SensorStatus(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class ImageEncoding(Enum):
    RGB8 = "rgb8"
    RGBA8 = "rgba8"
    BGR8 = "bgr8"
    MONO8 = "mono8"
    MONO16 = "mono16"
    FLOAT32 = "float32"


class PointCloudFormat(Enum):
    XYZ = "xyz"
    XYZI = "xyzi"          # + intensity
    XYZRGB = "xyzrgb"


class LidarKind(Enum):
    ROTATING = "rotating"
    SOLID_STATE = "solid_state"


class RadarKind(Enum):
    FMCW = "fmcw"


# ════════════════════════════════════════════════════════════════════════
# Typed sensor output
# ════════════════════════════════════════════════════════════════════════

@dataclass
class SensorData:
    """Base class for every typed sensor output. Every concrete output
    (ImageFrame, PointCloud, ...) carries these five fields plus its own
    modality-specific payload."""

    sensor_id: str
    frame_id: str
    timestamp: float
    sequence_number: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def base_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "sequence_number": self.sequence_number,
            "metadata": self.metadata,
        }


@dataclass
class ImageFrame(SensorData):
    width: int = 0
    height: int = 0
    channels: int = 3
    encoding: ImageEncoding = ImageEncoding.RGB8
    data: Optional[bytes] = None
    exposure: float = 0.0
    gamma: float = 1.0

    def to_dict(self) -> dict:
        d = self.base_dict()
        d.update({
            "width": self.width, "height": self.height, "channels": self.channels,
            "encoding": self.encoding.value, "has_data": self.data is not None,
            "exposure": self.exposure, "gamma": self.gamma,
        })
        return d


@dataclass
class DepthFrame(SensorData):
    width: int = 0
    height: int = 0
    near: float = 0.1
    far: float = 100.0
    encoding: str = "float32"
    depth_data: Optional[list[float]] = None
    point_cloud: Optional["PointCloud"] = None

    def to_dict(self) -> dict:
        d = self.base_dict()
        d.update({
            "width": self.width, "height": self.height, "near": self.near, "far": self.far,
            "encoding": self.encoding, "has_depth_data": self.depth_data is not None,
            "has_point_cloud": self.point_cloud is not None,
        })
        return d


@dataclass
class ThermalFrame(SensorData):
    width: int = 0
    height: int = 0
    temperature_data: Optional[list[float]] = None
    min_temp_k: float = 0.0
    max_temp_k: float = 0.0
    false_color: bool = False

    def to_dict(self) -> dict:
        d = self.base_dict()
        d.update({
            "width": self.width, "height": self.height,
            "has_temperature_data": self.temperature_data is not None,
            "min_temp_k": self.min_temp_k, "max_temp_k": self.max_temp_k,
            "false_color": self.false_color,
        })
        return d


@dataclass
class PointCloud(SensorData):
    points: list[tuple[float, ...]] = field(default_factory=list)  # (x,y,z[,i][,r,g,b])
    format: PointCloudFormat = PointCloudFormat.XYZ
    point_count: int = 0

    def __post_init__(self) -> None:
        if self.point_count == 0:
            self.point_count = len(self.points)

    def to_dict(self) -> dict:
        d = self.base_dict()
        d.update({"format": self.format.value, "point_count": self.point_count})
        return d


@dataclass
class RadarTarget:
    range_m: float
    velocity_ms: float
    azimuth_rad: float
    elevation_rad: float
    rcs_dbsm: float
    track_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RadarTargets(SensorData):
    targets: list[RadarTarget] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.base_dict()
        d["targets"] = [t.to_dict() for t in self.targets]
        return d


@dataclass
class GPSReading(SensorData):
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    altitude_m: float = 0.0
    accuracy_m: float = 0.0
    velocity_ms: Vec3 = field(default_factory=Vec3)
    heading_rad: float = 0.0

    def to_dict(self) -> dict:
        d = self.base_dict()
        d.update({
            "latitude_deg": self.latitude_deg, "longitude_deg": self.longitude_deg,
            "altitude_m": self.altitude_m, "accuracy_m": self.accuracy_m,
            "velocity_ms": self.velocity_ms.to_dict(), "heading_rad": self.heading_rad,
        })
        return d


@dataclass
class IMUReading(SensorData):
    linear_acceleration: Vec3 = field(default_factory=Vec3)
    angular_velocity: Vec3 = field(default_factory=Vec3)
    orientation: Quaternion = field(default_factory=Quaternion.identity)
    linear_acceleration_covariance: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 9)
    angular_velocity_covariance: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 9)
    orientation_covariance: tuple[float, ...] = field(default_factory=lambda: (0.0,) * 9)

    def to_dict(self) -> dict:
        d = self.base_dict()
        d.update({
            "linear_acceleration": self.linear_acceleration.to_dict(),
            "angular_velocity": self.angular_velocity.to_dict(),
            "orientation": self.orientation.to_dict(),
        })
        return d
