"""
base_sensor.py
══════════════════════════════════════════════════════════════════════════
Abstract base class for every sensor in the PhysWorldLM Sensors Framework.

A `Sensor` is a physics-aware, simulator-independent representation of a
real sensing device: it carries intrinsic/extrinsic parameters, mount
state, timing, noise, and calibration -- and knows how to serialize
itself -- but contains no simulator-specific code. Backend adapters
(Omniverse RTX sensors, Isaac Sim sensors, Gazebo, ROS2, MuJoCo, ...)
translate a `Sensor` into their own runtime objects; `Sensor` itself
never imports any of those.

Thread-safety: every `Sensor` guards its mutable lifecycle/state behind
one re-entrant lock (`self._lock`), matching the convention used by
`SensorManager` and the rest of PhysWorldLM's compiler stages, so a
sensor can safely be updated/captured from one thread while inspected
(`health()`, `serialize()`) from another.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from sensor_types import (
    CalibrationData,
    NoiseModel,
    SensorData,
    SensorStatus,
    SensorType,
    TimingModel,
    Transform6DoF,
)

__all__ = [
    "SensorLifecycleError",
    "SensorValidationError",
    "SensorHealth",
    "SensorDiagnostic",
    "DiagnosticSeverity",
    "Sensor",
]

logger = logging.getLogger("physworldlm.sensors")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ════════════════════════════════════════════════════════════════════════
# Exceptions
# ════════════════════════════════════════════════════════════════════════

class SensorLifecycleError(Exception):
    """Raised when a lifecycle method is invoked out of order
    (e.g. `capture()` before `start()`, `update()` after `stop()`)."""


class SensorValidationError(Exception):
    """Raised by `validate()` callers that choose to treat problems as fatal.
    `validate()` itself returns a list of problem strings rather than raising,
    so callers (SensorManager, tests) can decide policy."""


# ════════════════════════════════════════════════════════════════════════
# Diagnostics / health
# ════════════════════════════════════════════════════════════════════════

class DiagnosticSeverity(Enum):
    INFO = auto()
    WARNING = auto()
    ERROR = auto()


@dataclass
class SensorDiagnostic:
    severity: DiagnosticSeverity
    message: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"severity": self.severity.name, "message": self.message, "timestamp": self.timestamp}


@dataclass
class SensorHealth:
    """Snapshot of a sensor's operating health at a point in time."""

    status: SensorStatus
    last_update_timestamp: Optional[float]
    update_count: int
    error_count: int
    warning_count: int
    dropped_frame_count: int
    average_update_interval_s: Optional[float]
    recent_diagnostics: list[SensorDiagnostic] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "last_update_timestamp": self.last_update_timestamp,
            "update_count": self.update_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "dropped_frame_count": self.dropped_frame_count,
            "average_update_interval_s": self.average_update_interval_s,
            "recent_diagnostics": [d.to_dict() for d in self.recent_diagnostics],
        }


# ════════════════════════════════════════════════════════════════════════
# Sensor (abstract base)
# ════════════════════════════════════════════════════════════════════════

class Sensor(ABC):
    """Abstract, simulator-independent sensor.

    Subclasses (`Camera`, `Lidar`, `Radar`, `Imu`, `Gps`, `ThermalCamera`,
    `DepthCamera`, ...) add modality-specific fields and implement
    `capture()`. Everything else -- lifecycle, mounting, health,
    serialization -- lives here so every sensor behaves consistently.
    """

    # Overridden by subclasses.
    sensor_type: SensorType = SensorType.CAMERA

    _MAX_DIAGNOSTIC_HISTORY = 50

    def __init__(
        self,
        name: str,
        *,
        sensor_id: Optional[str] = None,
        parent_entity_id: Optional[str] = None,
        frame_id: Optional[str] = None,
        mount_transform: Optional[Transform6DoF] = None,
        update_rate_hz: float = 30.0,
        enabled: bool = True,
        noise_model: Optional[NoiseModel] = None,
        timing_model: Optional[TimingModel] = None,
        calibration: Optional[CalibrationData] = None,
        fov_deg: float = 60.0,
        near_clip: float = 0.01,
        far_clip: float = 1000.0,
        resolution: tuple[int, int] = (1, 1),
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.sensor_id: str = sensor_id or str(uuid.uuid4())
        self.name: str = name
        self.parent_entity_id: Optional[str] = parent_entity_id
        self.frame_id: str = frame_id or f"{name}_frame"
        self.mount_transform: Transform6DoF = mount_transform or Transform6DoF.identity()

        self.update_rate_hz: float = update_rate_hz
        self.enabled: bool = enabled
        self.timestamp: float = 0.0
        self.latency_s: float = 0.0

        self.noise_model: Optional[NoiseModel] = noise_model
        self.timing_model: Optional[TimingModel] = timing_model
        self.calibration: CalibrationData = calibration or CalibrationData()

        self.fov_deg: float = fov_deg
        self.near_clip: float = near_clip
        self.far_clip: float = far_clip
        self.resolution: tuple[int, int] = resolution

        self.metadata: dict[str, Any] = metadata or {}

        self._status: SensorStatus = SensorStatus.UNINITIALIZED
        self._lock = threading.RLock()
        self._sequence_number: int = 0
        self._update_count: int = 0
        self._error_count: int = 0
        self._warning_count: int = 0
        self._dropped_frame_count: int = 0
        self._last_update_timestamp: Optional[float] = None
        self._update_intervals: list[float] = []
        self._diagnostics: list[SensorDiagnostic] = []
        self._accumulated_time_s: float = 0.0

    # ── lifecycle ────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Prepare the sensor for operation. Idempotent."""
        with self._lock:
            if self._status not in (SensorStatus.UNINITIALIZED, SensorStatus.STOPPED, SensorStatus.ERROR):
                return
            self._status = SensorStatus.INITIALIZED
            self._log(DiagnosticSeverity.INFO, "Sensor initialized.")

    def start(self) -> None:
        """Begin active sensing. Requires `initialize()` first."""
        with self._lock:
            if self._status not in (SensorStatus.INITIALIZED, SensorStatus.STOPPED, SensorStatus.PAUSED):
                raise SensorLifecycleError(
                    f"Cannot start() sensor '{self.name}' from status '{self._status.value}'."
                )
            self._status = SensorStatus.RUNNING
            self._log(DiagnosticSeverity.INFO, "Sensor started.")

    def stop(self) -> None:
        """Halt sensing entirely. Requires `start()`/`initialize()` first."""
        with self._lock:
            self._status = SensorStatus.STOPPED
            self._log(DiagnosticSeverity.INFO, "Sensor stopped.")

    def pause(self) -> None:
        """Temporarily suspend updates without losing configuration/state."""
        with self._lock:
            if self._status is not SensorStatus.RUNNING:
                raise SensorLifecycleError(f"Cannot pause() sensor '{self.name}' that is not running.")
            self._status = SensorStatus.PAUSED
            self._log(DiagnosticSeverity.INFO, "Sensor paused.")

    def resume(self) -> None:
        """Resume from `pause()`."""
        with self._lock:
            if self._status is not SensorStatus.PAUSED:
                raise SensorLifecycleError(f"Cannot resume() sensor '{self.name}' that is not paused.")
            self._status = SensorStatus.RUNNING
            self._log(DiagnosticSeverity.INFO, "Sensor resumed.")

    def update(self, dt: float) -> None:
        """Advance internal time/state by `dt` seconds.

        Called every simulation tick regardless of `update_rate_hz`; the
        sensor itself decides (via `_should_sample`) whether this tick
        produces a new sample. Subclasses that need per-tick state
        evolution (e.g. IMU integrating bias/drift) override
        `_on_update` rather than this method.
        """
        with self._lock:
            if self._status is not SensorStatus.RUNNING or not self.enabled:
                return
            self._accumulated_time_s += dt
            self.timestamp += dt
            self._on_update(dt)
            now = time.time()
            if self._last_update_timestamp is not None:
                self._update_intervals.append(now - self._last_update_timestamp)
                if len(self._update_intervals) > 200:
                    self._update_intervals.pop(0)
            self._last_update_timestamp = now
            self._update_count += 1

    def _on_update(self, dt: float) -> None:
        """Hook for subclasses that need per-tick state evolution. No-op by default."""
        return None

    def _should_sample(self) -> bool:
        """Whether enough time has accumulated to emit a new sample at `update_rate_hz`."""
        if self.update_rate_hz <= 0:
            return True
        period = 1.0 / self.update_rate_hz
        if self._accumulated_time_s >= period:
            self._accumulated_time_s -= period
            return True
        return False

    def reset(self) -> None:
        """Reset all accumulated state (sequence numbers, noise state, timers)
        while preserving configuration (mount, calibration, rates)."""
        with self._lock:
            self.timestamp = 0.0
            self._sequence_number = 0
            self._update_count = 0
            self._error_count = 0
            self._warning_count = 0
            self._dropped_frame_count = 0
            self._last_update_timestamp = None
            self._update_intervals.clear()
            self._accumulated_time_s = 0.0
            if self.noise_model is not None:
                self.noise_model.reset()
            self._log(DiagnosticSeverity.INFO, "Sensor reset.")

    # ── capture ──────────────────────────────────────────────────────

    def capture(self, raw_data: Any = None) -> Optional[SensorData]:
        """Produce one typed `SensorData` sample.

        Applies the timing model (packet loss / frame drop / latency)
        and sequence bookkeeping uniformly, then delegates the
        modality-specific payload construction to `_build_sample()`
        (implemented by each subclass). Returns `None` if the timing
        model drops this frame, or if the sensor is disabled/not
        running.

        Args:
            raw_data: Optional backend-supplied payload (e.g. rendered
                pixels, a depth buffer) that a backend adapter passes in.
                The framework itself never renders; when `raw_data` is
                omitted the returned `SensorData` carries structurally
                valid metadata with an empty payload.
        """
        with self._lock:
            if not self.enabled or self._status not in (SensorStatus.RUNNING, SensorStatus.INITIALIZED):
                return None

            if self.timing_model is not None:
                if self.timing_model.should_drop():
                    self._dropped_frame_count += 1
                    self._log(DiagnosticSeverity.WARNING, "Frame dropped by timing model.")
                    return None
                self.latency_s = self.timing_model.next_delay_s()

            sample = self._build_sample(raw_data)
            self._sequence_number += 1
            return sample

    @abstractmethod
    def _build_sample(self, raw_data: Any) -> SensorData:
        """Construct this sensor's typed output. Implemented by each subclass."""

    def _next_data_kwargs(self) -> dict:
        """Common `SensorData` fields every subclass's `_build_sample` should include."""
        return {
            "sensor_id": self.sensor_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "sequence_number": self._sequence_number,
            "metadata": {"latency_s": self.latency_s, "sensor_name": self.name},
        }

    # ── mounting ─────────────────────────────────────────────────────

    def attach(self, parent_entity_id: str, mount_transform: Optional[Transform6DoF] = None) -> None:
        """Attach this sensor to `parent_entity_id` at an optional mount offset."""
        with self._lock:
            self.parent_entity_id = parent_entity_id
            if mount_transform is not None:
                self.mount_transform = mount_transform
            self._log(DiagnosticSeverity.INFO, f"Attached to entity '{parent_entity_id}'.")

    def detach(self) -> None:
        """Detach this sensor from its parent entity."""
        with self._lock:
            previous = self.parent_entity_id
            self.parent_entity_id = None
            self._log(DiagnosticSeverity.INFO, f"Detached from entity '{previous}'.")

    def world_transform(self, parent_world_transform: Transform6DoF) -> Transform6DoF:
        """Compose the parent entity's world transform with this sensor's
        mount offset to get the sensor's world-space pose."""
        return parent_world_transform.compose(self.mount_transform)

    # ── validation ───────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Return a list of human-readable problems with this sensor's
        current configuration. Empty list means valid. Never raises."""
        problems: list[str] = []
        if not self.name:
            problems.append("Sensor name must be non-empty.")
        if self.update_rate_hz < 0:
            problems.append(f"update_rate_hz must be >= 0 (got {self.update_rate_hz}).")
        if self.near_clip < 0:
            problems.append(f"near_clip must be >= 0 (got {self.near_clip}).")
        if self.far_clip <= self.near_clip:
            problems.append(f"far_clip ({self.far_clip}) must be > near_clip ({self.near_clip}).")
        if self.fov_deg <= 0 or self.fov_deg > 360:
            problems.append(f"fov_deg must be in (0, 360] (got {self.fov_deg}).")
        if any(dim <= 0 for dim in self.resolution):
            problems.append(f"resolution dimensions must be > 0 (got {self.resolution}).")
        problems.extend(self._validate_specific())
        return problems

    def _validate_specific(self) -> list[str]:
        """Hook for subclass-specific validation. No-op by default."""
        return []

    # ── health / diagnostics ────────────────────────────────────────

    def _log(self, severity: DiagnosticSeverity, message: str) -> None:
        diag = SensorDiagnostic(severity=severity, message=message)
        self._diagnostics.append(diag)
        if len(self._diagnostics) > self._MAX_DIAGNOSTIC_HISTORY:
            self._diagnostics.pop(0)
        if severity is DiagnosticSeverity.WARNING:
            self._warning_count += 1
            logger.warning("[%s] %s", self.name, message)
        elif severity is DiagnosticSeverity.ERROR:
            self._error_count += 1
            logger.error("[%s] %s", self.name, message)
        else:
            logger.debug("[%s] %s", self.name, message)

    def health(self) -> SensorHealth:
        with self._lock:
            avg_interval = (
                sum(self._update_intervals) / len(self._update_intervals)
                if self._update_intervals else None
            )
            return SensorHealth(
                status=self._status,
                last_update_timestamp=self._last_update_timestamp,
                update_count=self._update_count,
                error_count=self._error_count,
                warning_count=self._warning_count,
                dropped_frame_count=self._dropped_frame_count,
                average_update_interval_s=avg_interval,
                recent_diagnostics=list(self._diagnostics[-10:]),
            )

    @property
    def status(self) -> SensorStatus:
        return self._status

    # ── serialization ───────────────────────────────────────────────

    def serialize(self) -> dict:
        """JSON-serializable representation of this sensor's full
        configuration and lifecycle state (not its captured data)."""
        with self._lock:
            return {
                "sensor_id": self.sensor_id,
                "name": self.name,
                "sensor_type": self.sensor_type.value,
                "parent_entity_id": self.parent_entity_id,
                "frame_id": self.frame_id,
                "mount_transform": self.mount_transform.to_dict(),
                "update_rate_hz": self.update_rate_hz,
                "enabled": self.enabled,
                "timestamp": self.timestamp,
                "latency_s": self.latency_s,
                "noise_model": self.noise_model.to_dict() if self.noise_model else None,
                "timing_model": self.timing_model.to_dict() if self.timing_model else None,
                "calibration": self.calibration.to_dict(),
                "fov_deg": self.fov_deg,
                "near_clip": self.near_clip,
                "far_clip": self.far_clip,
                "resolution": list(self.resolution),
                "metadata": self.metadata,
                "status": self._status.value,
                **self._serialize_specific(),
            }

    def _serialize_specific(self) -> dict:
        """Hook for subclass-specific fields. No-op by default."""
        return {}

    @classmethod
    def deserialize(cls, data: dict) -> "Sensor":
        """Reconstruct a sensor from `serialize()` output.

        Base implementation restores the common fields; subclasses
        override to additionally restore modality-specific fields,
        calling `super().deserialize(data)`-equivalent construction via
        their own `__init__`.
        """
        instance = cls(
            name=data["name"],
            sensor_id=data.get("sensor_id"),
            parent_entity_id=data.get("parent_entity_id"),
            frame_id=data.get("frame_id"),
            mount_transform=Transform6DoF.from_dict(data.get("mount_transform", {})),
            update_rate_hz=data.get("update_rate_hz", 30.0),
            enabled=data.get("enabled", True),
            calibration=CalibrationData.from_dict(data.get("calibration", {})),
            fov_deg=data.get("fov_deg", 60.0),
            near_clip=data.get("near_clip", 0.01),
            far_clip=data.get("far_clip", 1000.0),
            resolution=tuple(data.get("resolution", (1, 1))),
            metadata=data.get("metadata", {}),
        )
        instance.timestamp = data.get("timestamp", 0.0)
        instance.latency_s = data.get("latency_s", 0.0)
        status_value = data.get("status")
        if status_value:
            instance._status = SensorStatus(status_value)
        return instance

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, sensor_id={self.sensor_id!r}, "
            f"status={self._status.value}, parent_entity_id={self.parent_entity_id!r})"
        )
