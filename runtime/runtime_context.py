"""
runtime_context.py
══════════════════════════════════════════════════════════════════════════
Shared runtime state for the PhysWorldLM Runtime Engine.

Pipeline position
------------------
                ...SceneCompiler (existing)
                        │
                        ▼
               CompilationReport / SceneGraph
                        │
                        ▼
        ┌──────────────────────────────────────────────────┐
        │              RUNTIME ENGINE                      │
        │                                                  │
        │   RuntimeContext  ◄───────────────┐              │
        │        │                          │              │
        │        ▼                          │              │
        │   Timeline   EntityManager   SimulationController│
        │                                                  │
        └──────────────────────────┬───────────────────────┘
                                   ▼
                          BackendAdapter (protocol)
                                   │
                                   ▼
                 Omniverse / Gazebo / MuJoCo / Unity / Unreal

Scope
-----
`RuntimeContext` is the *only* shared, mutable-state container the
runtime's collaborators (`Timeline`, `EntityManager`,
`SimulationController`) are threaded through. It plays the same role at
runtime that `world_pipeline.ExecutionContext` plays at compile time:
one place to find configuration, the event bus, the scene under
execution, and per-run statistics -- without any of those collaborators
reaching back into each other directly.

This module intentionally contains **no** simulator-specific imports.
`BackendAdapter` below is a narrow protocol; concrete adapters (an
Omniverse adapter, a Gazebo adapter, a MuJoCo adapter, ...) are supplied
by callers exactly the way `world_pipeline.SimulationBackend`
implementations are, and are invoked only through this protocol.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, Protocol, TYPE_CHECKING, runtime_checkable

from world_pipeline import Event, EventBus, EventType

if TYPE_CHECKING:
    from scene_compiler import SceneGraph
    from world_spec import WorldSpec


# ════════════════════════════════════════════════════════════════════════
# Exception hierarchy
# ════════════════════════════════════════════════════════════════════════

class RuntimeEngineError(Exception):
    """Base exception for every failure raised by the Runtime Engine."""


class RuntimeLifecycleError(RuntimeEngineError):
    """Raised when a runtime API is invoked out of lifecycle order."""


class RuntimeConfigurationError(RuntimeEngineError):
    """Raised when the runtime is misconfigured for the requested operation."""


class BackendAdapterError(RuntimeEngineError):
    """Raised when a `BackendAdapter` call fails or the backend is unavailable."""


class EntityError(RuntimeEngineError):
    """Base exception for `EntityManager` failures."""


class EntityNotFoundError(EntityError):
    """Raised when an entity lookup fails."""


class EntityAttachmentError(EntityError):
    """Raised on an invalid attach/detach/reparent operation (e.g. a cycle)."""


class TimelineError(RuntimeEngineError):
    """Base exception for `Timeline` failures."""


class RuntimeCancelledError(RuntimeEngineError):
    """Raised internally when a run is cancelled; never expected to escape the controller."""


class SnapshotError(RuntimeEngineError):
    """Raised when a checkpoint/snapshot/restore operation fails."""


# ════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════

class RuntimePhase(Enum):
    """Lifecycle phase of a `SimulationController`."""

    UNINITIALIZED = 0
    INITIALIZED = 1
    PLAYING = 2
    PAUSED = 3
    STOPPED = 4
    SHUTDOWN = 99

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


class RuntimeEventType(Enum):
    """Runtime-level events broadcast on the shared `EventBus`.

    Deliberately distinct from `world_pipeline.EventType`, which covers
    *compile-time* pipeline events -- these cover *execution-time* events,
    published on the same bus so a single logger/telemetry sink can
    observe both halves of PhysWorldLM without change.
    """

    RUNTIME_INITIALIZED = auto()
    RUNTIME_SHUTDOWN = auto()
    PLAY_STARTED = auto()
    PAUSED = auto()
    RESUMED = auto()
    STOPPED = auto()
    RESET = auto()
    STEP_COMPLETED = auto()
    ENTITY_CREATED = auto()
    ENTITY_DESTROYED = auto()
    ENTITY_ATTACHED = auto()
    ENTITY_DETACHED = auto()
    SNAPSHOT_SAVED = auto()
    SNAPSHOT_RESTORED = auto()
    BENCHMARK_ITERATION = auto()
    RUNTIME_ERROR = auto()


# ════════════════════════════════════════════════════════════════════════
# Narrow collaborator protocols
# ════════════════════════════════════════════════════════════════════════
# None of these are implemented in this module. They are invoked, never
# subclassed, matching the dependency-injection style already used by
# `world_pipeline.SimulationBackend` / `AssetResolver`.

@runtime_checkable
class BackendAdapter(Protocol):
    """The *only* seam through which the runtime touches a concrete
    simulator. `SimulationController` calls these methods and nothing
    else -- it never imports Omniverse, Gazebo, MuJoCo, Unity, Unreal,
    ROS, or any physics-engine API directly.
    """

    name: str

    def is_available(self) -> bool: ...

    def sync_entity_state(self, entities: "EntitySnapshotBatch") -> None: ...

    def step(self, dt: float) -> None: ...

    def read_back_state(self) -> "EntitySnapshotBatch": ...


@runtime_checkable
class SensorManagerProtocol(Protocol):
    """Extension seam for sensor execution (lidar, camera, IMU, ...).

    Mirrors `world_pipeline.SensorPlugin`'s "registrable now, invoked
    later" posture: the runtime threads this through `RuntimeContext` so
    a concrete sensor manager can be wired in without any of
    `Timeline`/`EntityManager`/`SimulationController` changing.
    """

    name: str

    def capture(self, dt: float) -> Any: ...


@runtime_checkable
class PhysicsStateProtocol(Protocol):
    """Opaque handle to whatever physics representation a backend adapter
    maintains. The runtime never inspects its contents -- only threads
    it through `RuntimeContext` for adapters/sensors that need it.
    """

    def to_dict(self) -> dict: ...


@dataclass(frozen=True)
class EntitySnapshotBatch:
    """A batch of per-entity transform/state snapshots exchanged between
    `EntityManager` and a `BackendAdapter` on each step.

    Kept intentionally generic (`payload` is a plain dict keyed by
    runtime entity id) so no backend-specific schema leaks into the
    runtime core.
    """

    frame: int
    payload: dict[str, dict[str, Any]] = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════

@dataclass
class RuntimeConfig:
    """User-configurable settings controlling `SimulationController` /
    `Timeline` / `EntityManager` behavior.

    Attributes:
        deterministic: If True, seeds and stepping are made
            reproducible; forwarded to `Timeline` and any `BackendAdapter`.
        fixed_timestep_s: Fixed simulation timestep, in seconds, used
            whenever `Timeline` is in `TimeMode.FIXED`.
        max_substeps: Upper bound on catch-up substeps `Timeline` will
            take in a single `advance()` call before it drops frames
            rather than spiraling.
        random_seed: If set, seeds Python's `random` module at
            `SimulationController.initialize()` for reproducibility.
        enable_profiling: If True, per-step and per-phase durations are
            recorded into `RuntimeStatistics`.
        enable_checkpointing: If True, `SimulationController` persists
            snapshots to `checkpoint_dir` at `checkpoint_interval_steps`.
        checkpoint_dir: Base directory for snapshot files.
        checkpoint_interval_steps: Snapshot cadence, in simulation steps.
        benchmark_mode: If True, `SimulationController.run_benchmark()`
            semantics apply (aggregate timing, no wall-clock throttling).
        real_time: If True, `Timeline.advance()` paces itself against
            wall-clock time; if False (typical for benchmarking/headless
            batch execution) steps run as fast as the backend allows.
        time_scale: Multiplier applied to wall-clock time when
            `real_time` is True (e.g. `2.0` = 2x speed).
        max_steps: Optional hard cap on total steps for a single
            `play()`/`run_benchmark()` invocation; `None` means unbounded
            (caller drives `stop()`/`pause()` externally).
    """

    deterministic: bool = True
    fixed_timestep_s: float = 1.0 / 60.0
    max_substeps: int = 8
    random_seed: Optional[int] = None
    enable_profiling: bool = True
    enable_checkpointing: bool = False
    checkpoint_dir: Optional[Path] = None
    checkpoint_interval_steps: int = 300
    benchmark_mode: bool = False
    real_time: bool = False
    time_scale: float = 1.0
    max_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.fixed_timestep_s <= 0:
            raise RuntimeConfigurationError("fixed_timestep_s must be > 0.")
        if self.max_substeps < 1:
            raise RuntimeConfigurationError("max_substeps must be >= 1.")


# ════════════════════════════════════════════════════════════════════════
# Statistics
# ════════════════════════════════════════════════════════════════════════

@dataclass
class RuntimeStatistics:
    """Quantitative summary of a runtime session, mirroring the shape of
    `world_pipeline.PipelineStatistics` so both halves of PhysWorldLM
    report diagnostics the same way.
    """

    total_wall_time_s: float = 0.0
    total_sim_time_s: float = 0.0
    total_steps: int = 0
    dropped_substeps: int = 0
    entities_created: int = 0
    entities_destroyed: int = 0
    snapshots_saved: int = 0
    snapshots_restored: int = 0
    error_count: int = 0
    mean_step_duration_s: float = 0.0
    max_step_duration_s: float = 0.0
    _step_duration_total_s: float = field(default=0.0, repr=False)

    def record_step(self, duration_s: float) -> None:
        self.total_steps += 1
        self._step_duration_total_s += duration_s
        self.mean_step_duration_s = self._step_duration_total_s / self.total_steps
        self.max_step_duration_s = max(self.max_step_duration_s, duration_s)

    def to_dict(self) -> dict:
        return {
            "total_wall_time_s": round(self.total_wall_time_s, 6),
            "total_sim_time_s": round(self.total_sim_time_s, 6),
            "total_steps": self.total_steps,
            "dropped_substeps": self.dropped_substeps,
            "entities_created": self.entities_created,
            "entities_destroyed": self.entities_destroyed,
            "snapshots_saved": self.snapshots_saved,
            "snapshots_restored": self.snapshots_restored,
            "error_count": self.error_count,
            "mean_step_duration_s": round(self.mean_step_duration_s, 6),
            "max_step_duration_s": round(self.max_step_duration_s, 6),
        }


@dataclass
class BenchmarkRuntimeStatistics:
    """Aggregate statistics across the iterations of a
    `SimulationController.run_benchmark()` call.
    """

    iterations: int = 0
    successes: int = 0
    failures: int = 0
    total_time_s: float = 0.0
    mean_time_s: float = 0.0
    min_time_s: float = 0.0
    max_time_s: float = 0.0
    per_iteration_s: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "successes": self.successes,
            "failures": self.failures,
            "total_time_s": round(self.total_time_s, 6),
            "mean_time_s": round(self.mean_time_s, 6),
            "min_time_s": round(self.min_time_s, 6),
            "max_time_s": round(self.max_time_s, 6),
            "per_iteration_s": [round(t, 6) for t in self.per_iteration_s],
        }


# ════════════════════════════════════════════════════════════════════════
# RuntimeContext
# ════════════════════════════════════════════════════════════════════════

@dataclass
class RuntimeContext:
    """Immutable-ish shared state threaded through every runtime
    collaborator, exactly mirroring the role `world_pipeline.ExecutionContext`
    plays at compile time.

    `Timeline`, `EntityManager`, and `SimulationController` all read
    configuration, the event bus, and the scene under execution from
    here; none of them holds a private reference to another collaborator
    -- they are handed exactly what they need by `SimulationController`
    at call time, keeping each independently unit-testable.

    Attributes:
        scene_graph: The compiled `SceneGraph` under execution (from
            `worldspec_builder.WorldSpecBuilder` or `scene_compiler.SceneCompiler`).
            `None` only in `SIMULATION_ONLY`-style flows where a caller
            drives entities directly rather than from a compiled scene.
        world_spec: The originating `WorldSpec`, retained for
            provenance/diagnostics; the runtime never re-derives from it.
        config: Runtime-wide settings.
        event_bus: Shared `EventBus`. Reused from `world_pipeline` rather
            than reimplemented, so pipeline and runtime events flow
            through one observability seam.
        backend: The active `BackendAdapter`, or `None` before
            `SimulationController.initialize()` binds one.
        sensor_manager: Optional `SensorManagerProtocol` collaborator.
        physics_state: Opaque, backend-owned physics state handle.
        statistics: Running `RuntimeStatistics` for the active session.
        scene_id: Convenience accessor for checkpoint/log naming.
        cancellation_event: Optional cooperative-cancellation token,
            checked at every step/phase boundary.
        metadata: Free-form bag for caller-supplied context (e.g. run id,
            experiment tags) that collaborators may read but never require.
    """

    config: RuntimeConfig
    event_bus: EventBus
    scene_graph: Optional["SceneGraph"] = None
    world_spec: Optional["WorldSpec"] = None
    backend: Optional[BackendAdapter] = None
    sensor_manager: Optional[SensorManagerProtocol] = None
    physics_state: Optional[PhysicsStateProtocol] = None
    statistics: RuntimeStatistics = field(default_factory=RuntimeStatistics)
    scene_id: str = "unnamed_scene"
    cancellation_event: Optional[threading.Event] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def publish(self, event_type: RuntimeEventType, **payload: Any) -> Event:
        """Broadcast a runtime event on the shared `EventBus`.

        Runtime events are tagged `payload["runtime_event"] = event_type.name`
        and published under `world_pipeline.EventType.DIAGNOSTIC` so they
        flow through the *same* wildcard subscription a pipeline-level
        logger/profiler/telemetry sink already uses, with no new bus or
        listener contract required.
        """
        payload = {**payload, "runtime_event": event_type.name}
        return self.event_bus.publish(EventType.DIAGNOSTIC, **payload)

    def is_cancelled(self) -> bool:
        return self.cancellation_event is not None and self.cancellation_event.is_set()

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "config": {
                "deterministic": self.config.deterministic,
                "fixed_timestep_s": self.config.fixed_timestep_s,
                "real_time": self.config.real_time,
                "time_scale": self.config.time_scale,
                "benchmark_mode": self.config.benchmark_mode,
            },
            "backend": getattr(self.backend, "name", None),
            "statistics": self.statistics.to_dict(),
        }


def new_runtime_context(
    *,
    config: Optional[RuntimeConfig] = None,
    event_bus: Optional[EventBus] = None,
    scene_graph: Optional["SceneGraph"] = None,
    world_spec: Optional["WorldSpec"] = None,
    scene_id: Optional[str] = None,
) -> RuntimeContext:
    """Convenience factory: build a `RuntimeContext` with sensible
    defaults, deriving `scene_id` from `world_spec` when available.
    """
    resolved_scene_id = scene_id or getattr(world_spec, "scene_id", None) or "unnamed_scene"
    return RuntimeContext(
        config=config or RuntimeConfig(),
        event_bus=event_bus or EventBus(),
        scene_graph=scene_graph,
        world_spec=world_spec,
        scene_id=resolved_scene_id,
    )


__all__ = [
    "RuntimeEngineError",
    "RuntimeLifecycleError",
    "RuntimeConfigurationError",
    "BackendAdapterError",
    "EntityError",
    "EntityNotFoundError",
    "EntityAttachmentError",
    "TimelineError",
    "RuntimeCancelledError",
    "SnapshotError",
    "RuntimePhase",
    "RuntimeEventType",
    "BackendAdapter",
    "SensorManagerProtocol",
    "PhysicsStateProtocol",
    "EntitySnapshotBatch",
    "RuntimeConfig",
    "RuntimeStatistics",
    "BenchmarkRuntimeStatistics",
    "RuntimeContext",
    "new_runtime_context",
]
