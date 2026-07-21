"""
sensor_manager.py
══════════════════════════════════════════════════════════════════════════
Central registry and orchestrator for every `Sensor` instance in a scene.

`SensorManager` owns no simulation logic of its own -- it is the same
kind of thin, dependency-checked orchestrator as `scene_compiler.py`'s
`BuilderRegistry` and `worldspec_builder.py`'s `WorldSpecBuilder`, one
level down the stack: creation/destruction, attach/detach, frequency-
gated scheduling, time synchronization, and health/diagnostics
aggregation across a whole sensor population.

Thread-safety: one re-entrant lock guards the registry; individual
`Sensor` instances guard their own state independently (see
`base_sensor.py`), so `update_all()`/`capture_all()` can safely run
concurrently with a single sensor being inspected from another thread.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

from base_sensor import Sensor, SensorHealth
from sensor_types import SensorData, SensorStatus, SensorType, Transform6DoF

if TYPE_CHECKING:
    from camera import Camera
    from depth_camera import DepthCamera
    from gps import Gps
    from imu import Imu
    from lidar import Lidar
    from radar import Radar
    from thermal import ThermalCamera

__all__ = [
    "SensorManagerError",
    "DuplicateSensorIdError",
    "UnknownSensorError",
    "SensorManagerStatistics",
    "SensorManagerHealthReport",
    "SensorManager",
]

logger = logging.getLogger("physworldlm.sensors.manager")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ════════════════════════════════════════════════════════════════════════
# Exceptions
# ════════════════════════════════════════════════════════════════════════

class SensorManagerError(Exception):
    """Base exception for `SensorManager` failures."""


class DuplicateSensorIdError(SensorManagerError):
    """Raised by `register()` when `sensor.sensor_id` is already registered."""


class UnknownSensorError(SensorManagerError):
    """Raised when an operation references a `sensor_id` not in the registry."""


# ════════════════════════════════════════════════════════════════════════
# Registry class -> constructor map (for `create()` / `from_world_spec`)
# ════════════════════════════════════════════════════════════════════════

def _default_sensor_classes() -> dict[str, type[Sensor]]:
    """Lazily import concrete sensor classes so `SensorManager` can be
    imported (and used purely as a container for externally-constructed
    `Sensor`s) without pulling in every concrete sensor module."""
    from camera import Camera
    from depth_camera import DepthCamera
    from gps import Gps
    from imu import Imu
    from lidar import Lidar
    from radar import Radar
    from thermal import ThermalCamera

    return {
        SensorType.CAMERA.value: Camera,
        SensorType.DEPTH_CAMERA.value: DepthCamera,
        SensorType.THERMAL_CAMERA.value: ThermalCamera,
        SensorType.LIDAR.value: Lidar,
        SensorType.RADAR.value: Radar,
        SensorType.GPS.value: Gps,
        SensorType.IMU.value: Imu,
    }


# ════════════════════════════════════════════════════════════════════════
# Statistics / health
# ════════════════════════════════════════════════════════════════════════

@dataclass
class SensorManagerStatistics:
    total_sensors: int = 0
    sensors_by_type: dict[str, int] = field(default_factory=dict)
    enabled_count: int = 0
    running_count: int = 0
    total_updates: int = 0
    total_captures: int = 0
    total_dropped_frames: int = 0
    total_errors: int = 0
    total_warnings: int = 0

    def to_dict(self) -> dict:
        return {
            "total_sensors": self.total_sensors,
            "sensors_by_type": self.sensors_by_type,
            "enabled_count": self.enabled_count,
            "running_count": self.running_count,
            "total_updates": self.total_updates,
            "total_captures": self.total_captures,
            "total_dropped_frames": self.total_dropped_frames,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
        }


@dataclass
class SensorManagerHealthReport:
    per_sensor: dict[str, SensorHealth]
    unhealthy_sensor_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "per_sensor": {sid: h.to_dict() for sid, h in self.per_sensor.items()},
            "unhealthy_sensor_ids": self.unhealthy_sensor_ids,
        }


# ════════════════════════════════════════════════════════════════════════
# SensorManager
# ════════════════════════════════════════════════════════════════════════

class SensorManager:
    """Registry + orchestrator for a scene's sensor population.

    Example:
        >>> manager = SensorManager()
        >>> cam = Camera("front_rgb", resolution=(1920, 1080))
        >>> manager.register(cam)
        >>> manager.attach_sensor(cam.sensor_id, parent_entity_id="tank_01")
        >>> manager.update_all(dt=1.0 / 60.0)
        >>> frames = manager.capture_all()
    """

    def __init__(self, sensor_classes: Optional[dict[str, type[Sensor]]] = None) -> None:
        self._sensors: dict[str, Sensor] = {}
        self._entity_index: dict[str, set[str]] = {}
        self._lock = threading.RLock()
        self._sensor_classes: dict[str, type[Sensor]] = dict(sensor_classes) if sensor_classes else {}
        self._sim_time_s: float = 0.0

    # ── class registry (for construction by type name) ────────────────

    def _resolve_sensor_classes(self) -> dict[str, type[Sensor]]:
        if not self._sensor_classes:
            self._sensor_classes = _default_sensor_classes()
        return self._sensor_classes

    def register_sensor_class(self, type_name: str, sensor_class: type[Sensor]) -> None:
        """Register a custom/extension sensor class under `type_name` so
        `create()`/`deserialize()` can construct it by name."""
        with self._lock:
            self._resolve_sensor_classes()
            self._sensor_classes[type_name] = sensor_class

    # ── register / unregister / create / destroy ──────────────────────

    def register(self, sensor: Sensor) -> Sensor:
        """Add an already-constructed `Sensor` to the registry."""
        with self._lock:
            if sensor.sensor_id in self._sensors:
                raise DuplicateSensorIdError(f"Sensor id '{sensor.sensor_id}' is already registered.")
            self._sensors[sensor.sensor_id] = sensor
            if sensor.parent_entity_id:
                self._entity_index.setdefault(sensor.parent_entity_id, set()).add(sensor.sensor_id)
            logger.info("Registered sensor '%s' (%s, id=%s).", sensor.name, sensor.sensor_type.value, sensor.sensor_id)
            return sensor

    def unregister(self, sensor_id: str) -> None:
        """Remove a sensor from the registry (does not call `stop()`; callers
        that need a clean shutdown should call `sensor.stop()` first)."""
        with self._lock:
            sensor = self._sensors.pop(sensor_id, None)
            if sensor is None:
                raise UnknownSensorError(f"No sensor registered with id '{sensor_id}'.")
            if sensor.parent_entity_id and sensor.parent_entity_id in self._entity_index:
                self._entity_index[sensor.parent_entity_id].discard(sensor_id)
                if not self._entity_index[sensor.parent_entity_id]:
                    del self._entity_index[sensor.parent_entity_id]
            logger.info("Unregistered sensor '%s' (id=%s).", sensor.name, sensor_id)

    def create(self, type_name: str, name: str, **kwargs: Any) -> Sensor:
        """Construct a sensor of `type_name` (e.g. "camera", "lidar") and
        register it in one call."""
        classes = self._resolve_sensor_classes()
        if type_name not in classes:
            raise SensorManagerError(
                f"Unknown sensor type '{type_name}'. Known types: {sorted(classes)}. "
                "Register custom types via register_sensor_class()."
            )
        sensor = classes[type_name](name, **kwargs)
        return self.register(sensor)

    def destroy(self, sensor_id: str) -> None:
        """Stop (if running) and unregister a sensor."""
        with self._lock:
            sensor = self.get(sensor_id)
            try:
                sensor.stop()
            except Exception:  # noqa: BLE001 - best-effort stop during teardown
                pass
            self.unregister(sensor_id)

    # ── lookup ──────────────────────────────────────────────────────

    def get(self, sensor_id: str) -> Sensor:
        with self._lock:
            sensor = self._sensors.get(sensor_id)
            if sensor is None:
                raise UnknownSensorError(f"No sensor registered with id '{sensor_id}'.")
            return sensor

    def find(self, predicate: Callable[[Sensor], bool]) -> list[Sensor]:
        with self._lock:
            return [s for s in self._sensors.values() if predicate(s)]

    def find_by_entity(self, parent_entity_id: str) -> list[Sensor]:
        with self._lock:
            ids = self._entity_index.get(parent_entity_id, set())
            return [self._sensors[sid] for sid in ids]

    def find_by_type(self, sensor_type: SensorType) -> list[Sensor]:
        with self._lock:
            return [s for s in self._sensors.values() if s.sensor_type is sensor_type]

    def all_sensors(self) -> list[Sensor]:
        with self._lock:
            return list(self._sensors.values())

    def __len__(self) -> int:
        return len(self._sensors)

    def __contains__(self, sensor_id: str) -> bool:
        return sensor_id in self._sensors

    # ── attach / detach ─────────────────────────────────────────────

    def attach_sensor(
        self, sensor_id: str, parent_entity_id: str, mount_transform: Optional[Transform6DoF] = None
    ) -> None:
        with self._lock:
            sensor = self.get(sensor_id)
            if sensor.parent_entity_id and sensor.parent_entity_id in self._entity_index:
                self._entity_index[sensor.parent_entity_id].discard(sensor_id)
            sensor.attach(parent_entity_id, mount_transform)
            self._entity_index.setdefault(parent_entity_id, set()).add(sensor_id)

    def detach_sensor(self, sensor_id: str) -> None:
        with self._lock:
            sensor = self.get(sensor_id)
            if sensor.parent_entity_id and sensor.parent_entity_id in self._entity_index:
                self._entity_index[sensor.parent_entity_id].discard(sensor_id)
                if not self._entity_index[sensor.parent_entity_id]:
                    del self._entity_index[sensor.parent_entity_id]
            sensor.detach()

    # ── time synchronization / scheduling ──────────────────────────

    def update_all(self, dt: float) -> None:
        """Advance every registered sensor's internal clock by `dt`
        seconds. Sensor-level `update_rate_hz` gating happens inside
        each sensor via `_should_sample()`; this call always ticks every
        sensor so their `timestamp`s stay synchronized to one shared
        simulation clock."""
        with self._lock:
            self._sim_time_s += dt
            for sensor in self._sensors.values():
                try:
                    sensor.update(dt)
                except Exception as exc:  # noqa: BLE001
                    logger.error("update() failed for sensor '%s': %s", sensor.name, exc)

    def capture_all(self, raw_data_by_sensor_id: Optional[dict[str, Any]] = None) -> dict[str, SensorData]:
        """Capture one sample from every enabled, running sensor that is
        due to sample this tick. Returns a mapping of `sensor_id ->
        SensorData` (sensors that produced no sample this tick, whether
        because they aren't due yet or their timing model dropped the
        frame, are simply absent from the result)."""
        raw_data_by_sensor_id = raw_data_by_sensor_id or {}
        results: dict[str, SensorData] = {}
        with self._lock:
            for sensor_id, sensor in self._sensors.items():
                if not sensor._should_sample():  # noqa: SLF001 - manager is a trusted collaborator
                    continue
                try:
                    sample = sensor.capture(raw_data_by_sensor_id.get(sensor_id))
                except Exception as exc:  # noqa: BLE001
                    logger.error("capture() failed for sensor '%s': %s", sensor.name, exc)
                    continue
                if sample is not None:
                    results[sensor_id] = sample
        return results

    # ── bulk lifecycle ──────────────────────────────────────────────

    def initialize_all(self) -> None:
        self._for_each(lambda s: s.initialize())

    def start_all(self) -> None:
        self._for_each(lambda s: s.start())

    def pause_all(self) -> None:
        self._for_each(lambda s: s.pause() if s.status is SensorStatus.RUNNING else None)

    def resume_all(self) -> None:
        self._for_each(lambda s: s.resume() if s.status is SensorStatus.PAUSED else None)

    def stop_all(self) -> None:
        self._for_each(lambda s: s.stop())

    def reset_all(self) -> None:
        self._sim_time_s = 0.0
        self._for_each(lambda s: s.reset())

    def _for_each(self, fn: Callable[[Sensor], None]) -> None:
        with self._lock:
            for sensor in self._sensors.values():
                try:
                    fn(sensor)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Bulk lifecycle operation failed for sensor '%s': %s", sensor.name, exc)

    # ── diagnostics ──────────────────────────────────────────────────

    def statistics(self) -> SensorManagerStatistics:
        with self._lock:
            stats = SensorManagerStatistics(total_sensors=len(self._sensors))
            for sensor in self._sensors.values():
                stats.sensors_by_type[sensor.sensor_type.value] = (
                    stats.sensors_by_type.get(sensor.sensor_type.value, 0) + 1
                )
                if sensor.enabled:
                    stats.enabled_count += 1
                if sensor.status is SensorStatus.RUNNING:
                    stats.running_count += 1
                health = sensor.health()
                stats.total_updates += health.update_count
                stats.total_dropped_frames += health.dropped_frame_count
                stats.total_errors += health.error_count
                stats.total_warnings += health.warning_count
            return stats

    def health(self) -> SensorManagerHealthReport:
        with self._lock:
            per_sensor = {sid: s.health() for sid, s in self._sensors.items()}
            unhealthy = [
                sid for sid, h in per_sensor.items()
                if h.status is SensorStatus.ERROR or h.error_count > 0
            ]
            return SensorManagerHealthReport(per_sensor=per_sensor, unhealthy_sensor_ids=unhealthy)

    def validate_all(self) -> dict[str, list[str]]:
        """Return `{sensor_id: [problems]}` for every sensor with validation
        problems. Empty dict means every sensor is valid."""
        with self._lock:
            problems = {sid: s.validate() for sid, s in self._sensors.items()}
            return {sid: p for sid, p in problems.items() if p}

    # ── serialization ────────────────────────────────────────────────

    def serialize(self) -> dict:
        with self._lock:
            return {
                "sim_time_s": self._sim_time_s,
                "sensors": [s.serialize() for s in self._sensors.values()],
            }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.serialize(), indent=indent)

    @classmethod
    def deserialize(cls, data: dict, sensor_classes: Optional[dict[str, type[Sensor]]] = None) -> "SensorManager":
        manager = cls(sensor_classes=sensor_classes)
        classes = manager._resolve_sensor_classes()
        for sensor_dict in data.get("sensors", []):
            type_name = sensor_dict.get("sensor_type")
            sensor_class = classes.get(type_name)
            if sensor_class is None:
                logger.warning("Skipping sensor with unknown type '%s' during deserialize().", type_name)
                continue
            sensor = sensor_class.deserialize(sensor_dict)
            manager.register(sensor)
        manager._sim_time_s = data.get("sim_time_s", 0.0)
        return manager

    # ── WorldSpec integration ───────────────────────────────────────

    @classmethod
    def from_world_spec(cls, world_spec: Any, sensor_spec_key: str = "sensors") -> "SensorManager":
        """Construct a `SensorManager` directly from a `WorldSpec`-like
        object.

        Defensive by design: the current `world_spec.WorldSpec` /
        `Entity` contract (see `world_spec.py`) does not yet define a
        per-entity sensor payload, so this reads an optional,
        forward-compatible location -- `entity.metadata[sensor_spec_key]`,
        a list of `{"type": ..., "name": ..., **sensor_kwargs}` dicts --
        and is a safe no-op (returns an empty manager) when that key is
        absent everywhere. This is the integration point for a future
        `SensorSpec` addition to the WorldSpec contract; no change to
        this framework's sensor classes is required when that lands.

        Args:
            world_spec: An object exposing `.entities`, each with `.id`
                and `.metadata` (duck-typed rather than importing
                `world_spec.WorldSpec` directly, to keep this framework
                dependency-free of the rest of PhysWorldLM).
            sensor_spec_key: The `entity.metadata` key holding the list
                of sensor definitions for that entity.
        """
        manager = cls()
        entities = getattr(world_spec, "entities", [])
        for entity in entities:
            metadata = getattr(entity, "metadata", {}) or {}
            sensor_specs = metadata.get(sensor_spec_key, [])
            for spec in sensor_specs:
                spec = dict(spec)
                type_name = spec.pop("type", None)
                name = spec.pop("name", None)
                if not type_name or not name:
                    logger.warning(
                        "Skipping malformed sensor spec on entity '%s': missing 'type' or 'name'.",
                        getattr(entity, "id", "<unknown>"),
                    )
                    continue
                try:
                    sensor = manager.create(type_name, name, **spec)
                except SensorManagerError as exc:
                    logger.warning("Skipping sensor spec on entity '%s': %s", getattr(entity, "id", "<unknown>"), exc)
                    continue
                manager.attach_sensor(sensor.sensor_id, getattr(entity, "id"))
        return manager

    def __repr__(self) -> str:
        return f"SensorManager(sensors={len(self._sensors)}, entities={len(self._entity_index)})"
